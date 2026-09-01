// Capability + acceleration-structure sizing probe. Needs the macOS 26 SDK (MTLGPUFamilyApple10 / Metal4).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

static const char *b(BOOL v) { return v ? "YES" : "NO "; }

int main(void) {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    printf("device                          = %s\n", [[d name] UTF8String]);
    printf("registryID                      = %llu\n", (unsigned long long)d.registryID);
    printf("hasUnifiedMemory                = %s\n", b(d.hasUnifiedMemory));
    printf("recommendedMaxWorkingSetSize    = %.2f GB\n", d.recommendedMaxWorkingSetSize / 1073741824.0);
    printf("maxThreadgroupMemoryLength      = %lu\n", (unsigned long)d.maxThreadgroupMemoryLength);
    printf("\n-- GPU families --\n");
    struct { const char *n; MTLGPUFamily f; } fams[] = {
      {"Apple6 ", MTLGPUFamilyApple6}, {"Apple7 ", MTLGPUFamilyApple7},
      {"Apple8 ", MTLGPUFamilyApple8}, {"Apple9 ", MTLGPUFamilyApple9},
      {"Apple10", MTLGPUFamilyApple10},
      {"Mac2   ", MTLGPUFamilyMac2},
      {"Metal3 ", MTLGPUFamilyMetal3}, {"Metal4 ", MTLGPUFamilyMetal4},
    };
    for (unsigned i = 0; i < sizeof(fams)/sizeof(fams[0]); i++)
      printf("  supportsFamily(%s)        = %s\n", fams[i].n, b([d supportsFamily:fams[i].f]));

    printf("\n-- ray tracing --\n");
    printf("supportsRaytracing              = %s\n", b(d.supportsRaytracing));
    printf("supportsRaytracingFromRender    = %s\n", b(d.supportsRaytracingFromRender));
    printf("supportsPrimitiveMotionBlur     = %s\n", b(d.supportsPrimitiveMotionBlur));
    printf("supportsFunctionPointers        = %s\n", b(d.supportsFunctionPointers));
    printf("supportsFunctionPointersFromRender = %s\n", b(d.supportsFunctionPointersFromRender));
    printf("supportsShaderBarycentricCoordinates = %s\n", b(d.supportsShaderBarycentricCoordinates));

    printf("\n-- MetalFX --\n");
    printf("MTLFXSpatialScaler supportsDevice           = %s\n",
           b([MTLFXSpatialScalerDescriptor supportsDevice:d]));
    printf("MTLFXTemporalScaler supportsDevice          = %s\n",
           b([MTLFXTemporalScalerDescriptor supportsDevice:d]));
    if (@available(macOS 26.0, *)) {
      printf("MTLFXTemporalDenoisedScaler supportsDevice  = %s\n",
             b([MTLFXTemporalDenoisedScalerDescriptor supportsDevice:d]));
      printf("MTLFXFrameInterpolator supportsDevice       = %s\n",
             b([MTLFXFrameInterpolatorDescriptor supportsDevice:d]));
    }

    // Acceleration structure sizing probe: 1 BLAS of N triangles.
    printf("\n-- acceleration structure sizing (triangle BLAS) --\n");
    for (int tri = 1000; tri <= 1000000; tri *= 10) {
      MTLAccelerationStructureTriangleGeometryDescriptor *g =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
      g.triangleCount = tri;
      g.vertexStride = 12;
      g.vertexFormat = MTLAttributeFormatFloat3;
      MTLPrimitiveAccelerationStructureDescriptor *pd =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
      pd.geometryDescriptors = @[g];
      pd.usage = MTLAccelerationStructureUsagePreferFastIntersection;
      MTLAccelerationStructureSizes s = [d accelerationStructureSizesWithDescriptor:pd];
      MTLSizeAndAlign sa = [d heapAccelerationStructureSizeAndAlignWithDescriptor:pd];
      printf("  %8d tris: AS=%8.3f MB  scratch=%8.3f MB  refitScratch=%8.3f MB  heapAlign=%lu\n",
             tri, s.accelerationStructureSize/1048576.0, s.buildScratchBufferSize/1048576.0,
             s.refitScratchBufferSize/1048576.0, (unsigned long)sa.align);
    }
    printf("\nargBufferTier                   = %ld\n", (long)d.argumentBuffersSupport);
  }
  return 0;
}
