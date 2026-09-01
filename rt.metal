#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct U {
    float3 eye;  float pad0;
    float3 fwd;  float pad1;
    float3 right;float pad2;
    float3 up;
    uint w; uint h; uint mode;
};

static inline float hashf(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return float(x) * 2.3283064e-10f;
}

kernel void trace(uint2 tid [[thread_position_in_grid]],
                  primitive_acceleration_structure accel [[buffer(0)]],
                  device uint *out [[buffer(1)]],
                  constant U &u [[buffer(2)]])
{
    if (tid.x >= u.w || tid.y >= u.h) return;
    uint seed = tid.y * u.w + tid.x;
    float2 uv = (float2(tid) + 0.5f) / float2(u.w, u.h) * 2.0f - 1.0f;
    uv.x *= float(u.w) / float(u.h);
    uint hit = 0;

    if (u.mode == 2) {
        // Interior: every ray starts at the centre and must hit the shell => 100% hit rate,
        // full traversal to a leaf, no early miss-exit at the root box.
        float z = hashf(seed * 2u + 1u) * 2.0f - 1.0f;
        float a = hashf(seed * 7919u) * 6.2831853f;
        float s = sqrt(max(0.0f, 1.0f - z * z));
        ray r;
        r.origin = float3(0.0f);
        r.direction = float3(s * cos(a), z, s * sin(a));
        r.min_distance = 0.0f; r.max_distance = 1e4f;
        intersector<triangle_data> it;
        auto res = it.intersect(r, accel);
        hit = (res.type != intersection_type::none) ? 1u : 0u;
    } else if (u.mode == 3) {
        // Primary closest-hit + 4 short any-hit AO rays from the hit point.
        ray r;
        r.origin = u.eye;
        r.direction = normalize(u.fwd + uv.x * u.right + uv.y * u.up);
        r.min_distance = 0.001f; r.max_distance = 1e4f;
        intersector<triangle_data> it;
        auto res = it.intersect(r, accel);
        if (res.type != intersection_type::none) {
            float3 p = r.origin + r.direction * res.distance;
            intersector<triangle_data> ao;
            ao.accept_any_intersection(true);
            for (uint i = 0; i < 4u; ++i) {
                float z = hashf(seed * 4u + i) * 2.0f - 1.0f;
                float a = hashf(seed * 131u + i * 17u) * 6.2831853f;
                float s = sqrt(max(0.0f, 1.0f - z * z));
                ray q;
                q.origin = p;
                q.direction = float3(s * cos(a), z, s * sin(a));
                q.min_distance = 0.01f; q.max_distance = 0.5f;
                auto rr = ao.intersect(q, accel);
                if (rr.type != intersection_type::none) hit++;
            }
        }
    } else {
        ray r;
        r.origin = u.eye;
        r.direction = normalize(u.fwd + uv.x * u.right + uv.y * u.up);
        r.min_distance = 0.001f; r.max_distance = 1e4f;
        intersector<triangle_data> it;
        if (u.mode == 1) it.accept_any_intersection(true);
        auto res = it.intersect(r, accel);
        hit = (res.type != intersection_type::none) ? 1u : 0u;
    }
    out[tid.y * u.w + tid.x] = hit;
}

// TLAS arms: the interior workload (origin at the centre of instance 0, random directions, 100% hit
// rate) through an instance acceleration structure. mode 0 = intersector<>, mode 1 = intersection_query<>.
kernel void trace_tlas(uint2 tid [[thread_position_in_grid]],
                       instance_acceleration_structure accel [[buffer(0)]],
                       device uint *out [[buffer(1)]],
                       constant U &u [[buffer(2)]])
{
    if (tid.x >= u.w || tid.y >= u.h) return;
    uint seed = tid.y * u.w + tid.x;
    float z = hashf(seed * 2u + 1u) * 2.0f - 1.0f;
    float a = hashf(seed * 7919u) * 6.2831853f;
    float s = sqrt(max(0.0f, 1.0f - z * z));
    ray r;
    r.origin = float3(0.0f);
    r.direction = float3(s * cos(a), z, s * sin(a));
    r.min_distance = 0.0f; r.max_distance = 1e4f;
    uint hit = 0;
    if (u.mode == 0) {
        intersector<triangle_data, instancing> it;
        auto res = it.intersect(r, accel, 0xFFu);
        hit = (res.type != intersection_type::none) ? 1u : 0u;
    } else {
        // Default intersection_params: the geometry and instances are opaque, so next() is expected to
        // auto-commit and return false at once; iters counts how often it actually handed back a
        // candidate (written into bits 1.. of out so the host can verify the shape of the traversal).
        intersection_query<triangle_data, instancing> iq(r, accel, 0xFFu, intersection_params());
        uint iters = 0;
        while (iq.next()) {
            iq.commit_triangle_intersection();
            iters++;
        }
        hit = (iq.get_committed_intersection_type() != intersection_type::none) ? 1u : 0u;
        hit |= iters << 1;
    }
    out[tid.y * u.w + tid.x] = hit;
}
