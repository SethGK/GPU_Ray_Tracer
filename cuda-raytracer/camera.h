#ifndef CAMERA_H
#define CAMERA_H

#include "utils.h"
#include "ray.h"

class camera {
public:
    __host__ __device__ camera() {}
    __host__ __device__ camera(point3 lookfrom, point3 lookat, vec3 vup, float vfov, float aspect_ratio) {
        float theta = degrees_to_radians(vfov);
        float h = tanf(theta/2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - w;
    }

    __host__ __device__ ray get_ray(float s, float t) const {
        return ray(origin, lower_left_corner + s*horizontal + t*vertical - origin);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
};

#endif
