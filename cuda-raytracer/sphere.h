#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "material.h"

struct sphere {
    point3 center;
    float radius;
    int material_type; // MaterialType
    vec3 albedo;
    float fuzz;
    float ir;

    __host__ __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        vec3 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float half_b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius*radius;
        float discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;
        float sqrtd = sqrtf(discriminant);

        float root = (-half_b - sqrtd) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max) return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.material_type = material_type;
        rec.albedo = albedo;
        rec.fuzz = fuzz;
        rec.ir = ir;
        return true;
    }
};

#endif
