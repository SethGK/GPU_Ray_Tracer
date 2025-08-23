#ifndef MATERIAL_H
#define MATERIAL_H

#include "utils.h"
#include "ray.h"
#include "hittable.h"

enum MaterialType { LAMBERTIAN=0, METAL=1, DIELECTRIC=2 };

__host__ __device__ inline bool lambertian_scatter(const ray& r_in, const hit_record& rec, RNG& rng, color& attenuation, ray& scattered) {
    // Generate a random scatter direction on the hemisphere oriented along the normal
    float z = random_float(rng, -1.0f, 1.0f);
    float a = random_float(rng, 0.0f, 2.0f*pi);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));
    vec3 rand_dir(r*cosf(a), r*sinf(a), z);

    vec3 scatter_direction = rec.normal + rand_dir;

    // Guard against a random direction that is perfectly opposite the normal,
    // which would result in a zero-length vector.
    if (scatter_direction.length_squared() < 1e-12f) {
        scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, unit_vector(scatter_direction));
    attenuation = rec.albedo;
    return true;
}

__host__ __device__ inline bool metal_scatter(const ray& r_in, const hit_record& rec, RNG& rng, color& attenuation, ray& scattered) {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    // Fuzz
    float z = random_float(rng, -1.0f, 1.0f);
    float a = random_float(rng, 0.0f, 2.0f*pi);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));
    vec3 rand_dir(r*cosf(a), r*sinf(a), z);
    scattered = ray(rec.p, reflected + rec.fuzz * rand_dir);
    attenuation = rec.albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__host__ __device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

__host__ __device__ inline bool dielectric_scatter(const ray& r_in, const hit_record& rec, RNG& rng, color& attenuation, ray& scattered) {
    attenuation = color(1.0f, 1.0f, 1.0f);
    float refraction_ratio = rec.front_face ? (1.0f/rec.ir) : rec.ir;

    vec3 unit_direction = unit_vector(r_in.direction());
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    vec3 direction;

    if (cannot_refract || schlick(cos_theta, refraction_ratio) > random_float(rng)) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    }

    scattered = ray(rec.p, direction);
    return true;
}

#endif
