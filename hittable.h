#ifndef HITTABLE_H
#define HITTABLE_H

#include "utils.h"

struct hit_record {
    point3 p;
    vec3 normal;
    float t;
    bool front_face;
    int material_id; // 0: lambertian, 1: metal, 2: dielectric

    __host__ __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    __host__ __device__ virtual ~hittable() {}
    __host__ __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
