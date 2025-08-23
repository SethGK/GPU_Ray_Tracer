#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <limits>
#include <stdint.h>
#include <cuda_runtime.h>

static constexpr float infinity = std::numeric_limits<float>::infinity();
static constexpr float pi = 3.1415926535897932385f;

__host__ __device__ inline float degrees_to_radians(float degrees) { return degrees * pi / 180.0f; }

struct RNG {
    __host__ __device__ explicit RNG(uint32_t seed = 1u) : state(seed?seed:1u) {}
    __host__ __device__ inline uint32_t next_uint() { uint32_t x=state; x^=x<<13; x^=x>>17; x^=x<<5; state=x; return x; }
    __host__ __device__ inline float next_float() { return (next_uint() & 0xFFFFFF) / 16777216.0f; }
    uint32_t state;
};

__host__ __device__ inline float random_float(RNG& rng) { return rng.next_float(); }
__host__ __device__ inline float random_float(RNG& rng, float min, float max) { return min + (max-min)*rng.next_float(); }

#endif
