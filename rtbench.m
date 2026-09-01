// Metal hardware ray-tracing microbenchmark (first measured on an Apple M5 Pro, macOS 26.5.2).
// Builds a triangle BLAS from a procedural noisy UV sphere, then times BLAS build under every usage
// flag, in-place refit, four ray workloads through the bare BLAS, and the same incoherent workload
// through a TLAS (1 and 16 instances) with both intersector<> and intersection_query<>.
// Timing = MTLCommandBuffer.GPUEndTime - GPUStartTime, median of 12 (build) / 40 (trace) runs.
// Usage: ./rtbench [SEG]   -> 2*SEG*SEG triangles (default 512 = 524,288)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

typedef struct { simd_float3 eye; float p0; simd_float3 fwd; float p1;
                 simd_float3 right; float p2; simd_float3 up; uint32_t w, h, mode; } U;

static int cmpd(const void *a, const void *b) {
  double x = *(const double*)a, y = *(const double*)b; return x<y?-1:(x>y?1:0);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    int SEG  = (argc > 1) ? atoi(argv[1]) : 512;   // -> 2*SEG*SEG triangles
    int RUNS = 40;
    int W = 1920, H = 1080;
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [d newCommandQueue];
    printf("device=%s  Apple10=%d  supportsRaytracing=%d  supportsRaytracingFromRender=%d\n",
           [[d name] UTF8String], (int)[d supportsFamily:MTLGPUFamilyApple10],
           (int)d.supportsRaytracing, (int)d.supportsRaytracingFromRender);

    // ---- procedural noisy UV sphere ----
    int rings = SEG, sectors = SEG;
    NSUInteger vcount = (NSUInteger)(rings + 1) * (sectors + 1);
    NSUInteger tcount = (NSUInteger)rings * sectors * 2;
    simd_float3 *verts = malloc(sizeof(simd_float3) * vcount);
    uint32_t *idx = malloc(sizeof(uint32_t) * tcount * 3);
    for (int i = 0; i <= rings; i++) {
      float phi = (float)M_PI * i / rings;
      for (int j = 0; j <= sectors; j++) {
        float th = 2.0f * (float)M_PI * j / sectors;
        float rr = 1.0f + 0.08f * sinf(7.0f*phi) * cosf(11.0f*th);
        verts[i*(sectors+1)+j] = (simd_float3){ rr*sinf(phi)*cosf(th), rr*cosf(phi), rr*sinf(phi)*sinf(th) };
      }
    }
    NSUInteger k = 0;
    for (int i = 0; i < rings; i++) for (int j = 0; j < sectors; j++) {
      uint32_t a = i*(sectors+1)+j, b = a+1, c = (i+1)*(sectors+1)+j, e = c+1;
      idx[k++]=a; idx[k++]=c; idx[k++]=b;  idx[k++]=b; idx[k++]=c; idx[k++]=e;
    }
    id<MTLBuffer> vb = [d newBufferWithBytes:verts length:sizeof(simd_float3)*vcount options:MTLResourceStorageModeShared];
    id<MTLBuffer> ib = [d newBufferWithBytes:idx  length:sizeof(uint32_t)*tcount*3 options:MTLResourceStorageModeShared];
    printf("mesh: %llu verts, %llu triangles\n", (unsigned long long)vcount, (unsigned long long)tcount);

    MTLAccelerationStructureTriangleGeometryDescriptor *g =
      [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    g.vertexBuffer = vb; g.vertexStride = sizeof(simd_float3);
    g.vertexFormat = MTLAttributeFormatFloat3;
    g.indexBuffer = ib; g.indexType = MTLIndexTypeUInt32;
    g.triangleCount = tcount;

    // ---- BLAS build under each usage flag ----
    struct { const char *name; MTLAccelerationStructureUsage u; } usages[] = {
      {"PreferFastIntersection", MTLAccelerationStructureUsagePreferFastIntersection},
      {"PreferFastBuild       ", MTLAccelerationStructureUsagePreferFastBuild},
      {"MinimizeMemory        ", MTLAccelerationStructureUsageMinimizeMemory},
      {"Refit                 ", MTLAccelerationStructureUsageRefit},
      {"None (default)        ", MTLAccelerationStructureUsageNone},
    };
    id<MTLAccelerationStructure> asFast = nil;
    printf("\n-- BLAS build, %llu tris --\n", (unsigned long long)tcount);
    for (unsigned ui = 0; ui < sizeof(usages)/sizeof(usages[0]); ui++) {
      MTLPrimitiveAccelerationStructureDescriptor *pd =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
      pd.geometryDescriptors = @[g];
      pd.usage = usages[ui].u;
      MTLAccelerationStructureSizes sz = [d accelerationStructureSizesWithDescriptor:pd];
      id<MTLAccelerationStructure> as = [d newAccelerationStructureWithSize:sz.accelerationStructureSize];
      id<MTLBuffer> scratch = [d newBufferWithLength:MAX(sz.buildScratchBufferSize,(NSUInteger)16)
                                             options:MTLResourceStorageModePrivate];
      double bt[12];
      for (int r = 0; r < 12; r++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
        [e buildAccelerationStructure:as descriptor:pd scratchBuffer:scratch scratchBufferOffset:0];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        bt[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
      }
      qsort(bt, 12, sizeof(double), cmpd);
      printf("  %s : build %6.3f ms (%6.1f Mtri/s) | AS %7.2f MB (%5.1f B/tri) | scratch %6.2f MB | refitScratch %6.2f MB\n",
             usages[ui].name, bt[6], tcount/(bt[6]/1000.0)/1e6,
             sz.accelerationStructureSize/1048576.0,
             (double)sz.accelerationStructureSize/(double)tcount,
             sz.buildScratchBufferSize/1048576.0, sz.refitScratchBufferSize/1048576.0);

      if (usages[ui].u == MTLAccelerationStructureUsagePreferFastIntersection) asFast = as;

      if (usages[ui].u == MTLAccelerationStructureUsageRefit) {
        id<MTLBuffer> rs = [d newBufferWithLength:MAX(sz.refitScratchBufferSize,(NSUInteger)16)
                                          options:MTLResourceStorageModePrivate];
        double rt[12];
        for (int r = 0; r < 12; r++) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
          [e refitAccelerationStructure:as descriptor:pd destination:nil scratchBuffer:rs scratchBufferOffset:0];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
          rt[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        }
        qsort(rt, 12, sizeof(double), cmpd);
        printf("  %s : REFIT %6.3f ms  (%.1fx cheaper than its own rebuild)\n",
               "                      ", rt[6], bt[6]/rt[6]);
      }
    }

    // ---- trace pipeline ----
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@"rt.metal" encoding:NSUTF8StringEncoding error:&err];
    if (!src) { printf("cannot read rt.metal: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLLibrary> lib = [d newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("SHADER ERROR: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"trace"] error:&err];
    if (!ps) { printf("PSO ERROR: %s\n", [[err description] UTF8String]); return 1; }
    printf("\npipeline: maxThreadsPerThreadgroup=%lu threadExecutionWidth=%lu\n",
           (unsigned long)ps.maxTotalThreadsPerThreadgroup, (unsigned long)ps.threadExecutionWidth);

    id<MTLBuffer> out = [d newBufferWithLength:(NSUInteger)W*H*4 options:MTLResourceStorageModePrivate];
    // Shared copy used for ONE untimed validation dispatch per workload, so the published timings keep the
    // private buffer and the hit rates are still measured, not assumed.
    id<MTLBuffer> val = [d newBufferWithLength:(NSUInteger)W*H*4 options:MTLResourceStorageModeShared];
    U u = {0};
    u.eye = (simd_float3){0,0,1.9f}; u.fwd = (simd_float3){0,0,-1};
    u.right = (simd_float3){1,0,0};  u.up = (simd_float3){0,1,0};
    u.w = W; u.h = H;

    const char *names[4] = {
      "CLOSEST-HIT primary (partial screen coverage)",
      "ANY-HIT     primary (partial screen coverage)",
      "CLOSEST-HIT interior, 100% hit rate          ",
      "PRIMARY + 4 short AO any-hit rays            " };
    double perpx[4] = {1,1,1,5};
    printf("\n-- ray throughput, %dx%d = %.2f Mpx --\n", W, H, W*H/1e6);
    for (int mode = 0; mode < 4; mode++) {
      u.mode = mode;
      double *t = malloc(sizeof(double)*RUNS);
      for (int r = 0; r < RUNS; r++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setAccelerationStructure:asFast atBufferIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e setBytes:&u length:sizeof(u) atIndex:2];
        [e dispatchThreads:MTLSizeMake(W,H,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        t[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
      }
      qsort(t, RUNS, sizeof(double), cmpd);
      {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setAccelerationStructure:asFast atBufferIndex:0];
        [e setBuffer:val offset:0 atIndex:1];
        [e setBytes:&u length:sizeof(u) atIndex:2];
        [e dispatchThreads:MTLSizeMake(W,H,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
      const uint32_t *v = val.contents; double sum = 0; long nz = 0;
      for (long i = 0; i < (long)W*H; i++) { sum += v[i]; nz += v[i] != 0; }
      double rays = (double)W*H*perpx[mode];
      printf("  %s : %6.3f ms median (%6.3f min) -> %7.1f Mray/s | %s %.1f%%%s\n",
             names[mode], t[RUNS/2], t[0], rays/(t[RUNS/2]/1000.0)/1e6,
             mode == 3 ? "px with primary+AO hits" : "hit rate", 100.0*nz/((double)W*H),
             mode == 3 ? [[NSString stringWithFormat:@", mean AO occluders/px %.2f", sum/((double)W*H)] UTF8String] : "");
      free(t);
    }

    // ---- TLAS arms: the same 100%-hit interior workload through an instance acceleration structure ----
    // (what an engine actually traces through), 1 instance and 16 instances of the same BLAS, with the
    // intersector<> object and with intersection_query<> (the ray-query form SPIRV-Cross emits for
    // GL_EXT_ray_query). Instances beyond the first sit outside the shell so the interior rays still
    // hit instance 0 -- the extra instances cost TLAS traversal, not hits.
    id<MTLComputePipelineState> pst = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"trace_tlas"] error:&err];
    if (!pst) { printf("PSO ERROR (trace_tlas): %s\n", [[err description] UTF8String]); return 1; }
    int counts[2] = { 1, 16 };
    printf("\n-- TLAS arms, interior 100%%-hit incoherent rays, %dx%d --\n", W, H);
    for (int ci = 0; ci < 2; ci++) {
      int n = counts[ci];
      id<MTLBuffer> inst = [d newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor) * n
                                          options:MTLResourceStorageModeShared];
      MTLAccelerationStructureInstanceDescriptor *desc = inst.contents;
      for (int i = 0; i < n; i++) {
        memset(&desc[i], 0, sizeof(desc[i]));
        // identity rotation; translate instance i>0 onto a ring of radius 4 (outside the unit shell).
        float ang = (float)i / (float)n * 6.2831853f;
        float tx = i ? 4.0f * cosf(ang) : 0.0f, tz = i ? 4.0f * sinf(ang) : 0.0f;
        desc[i].transformationMatrix.columns[0] = (MTLPackedFloat3){1, 0, 0};
        desc[i].transformationMatrix.columns[1] = (MTLPackedFloat3){0, 1, 0};
        desc[i].transformationMatrix.columns[2] = (MTLPackedFloat3){0, 0, 1};
        desc[i].transformationMatrix.columns[3] = (MTLPackedFloat3){tx, 0, tz};
        desc[i].options = MTLAccelerationStructureInstanceOptionOpaque;
        desc[i].mask = 0xFF;
        desc[i].accelerationStructureIndex = 0;
      }
      MTLInstanceAccelerationStructureDescriptor *td = [MTLInstanceAccelerationStructureDescriptor descriptor];
      td.instancedAccelerationStructures = @[asFast];
      td.instanceDescriptorBuffer = inst;
      td.instanceCount = n;
      MTLAccelerationStructureSizes tsz = [d accelerationStructureSizesWithDescriptor:td];
      id<MTLAccelerationStructure> tlas = [d newAccelerationStructureWithSize:tsz.accelerationStructureSize];
      id<MTLBuffer> tscratch = [d newBufferWithLength:MAX(tsz.buildScratchBufferSize,(NSUInteger)16)
                                              options:MTLResourceStorageModePrivate];
      double bt[12];
      for (int r = 0; r < 12; r++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
        [e useResource:asFast usage:MTLResourceUsageRead];
        [e buildAccelerationStructure:tlas descriptor:td scratchBuffer:tscratch scratchBufferOffset:0];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        bt[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
      }
      qsort(bt, 12, sizeof(double), cmpd);
      printf("  TLAS x%-2d build %6.3f ms | AS %6.1f KB | scratch %6.1f KB\n", n, bt[6],
             tsz.accelerationStructureSize / 1024.0, tsz.buildScratchBufferSize / 1024.0);
      const char *tnames[2] = { "intersector<triangle_data, instancing>      ",
                                "intersection_query<triangle_data, instancing>" };
      for (int mode = 0; mode < 2; mode++) {
        u.mode = mode;
        double *t = malloc(sizeof(double)*RUNS);
        for (int r = 0; r < RUNS; r++) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:pst];
          [e setAccelerationStructure:tlas atBufferIndex:0];
          [e useResource:asFast usage:MTLResourceUsageRead];
          [e setBuffer:out offset:0 atIndex:1];
          [e setBytes:&u length:sizeof(u) atIndex:2];
          [e dispatchThreads:MTLSizeMake(W,H,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
          t[r] = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        }
        qsort(t, RUNS, sizeof(double), cmpd);
        {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:pst];
          [e setAccelerationStructure:tlas atBufferIndex:0];
          [e useResource:asFast usage:MTLResourceUsageRead];
          [e setBuffer:val offset:0 atIndex:1];
          [e setBytes:&u length:sizeof(u) atIndex:2];
          [e dispatchThreads:MTLSizeMake(W,H,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        const uint32_t *v = val.contents; long hits = 0; double iters = 0;
        for (long i = 0; i < (long)W*H; i++) { hits += v[i] & 1u; iters += v[i] >> 1; }
        double rays = (double)W*H;
        printf("  TLAS x%-2d %s : %6.3f ms median (%6.3f min) -> %7.1f Mray/s | hit rate %.1f%%",
               n, tnames[mode], t[RUNS/2], t[0], rays/(t[RUNS/2]/1000.0)/1e6, 100.0*hits/rays);
        if (mode == 1) printf(", next() candidates/ray %.3f", iters/rays);
        printf("\n");
        free(t);
      }
    }
    free(verts); free(idx);
  }
  return 0;
}
