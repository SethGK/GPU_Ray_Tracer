#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "utils.h"
#include "camera.h"
#include "camera_params.h"
#include "sphere.h"

// Debug toggles disabled
#define DEBUG_FLAT 0
#define DEBUG_BG 0

__device__ inline float safe_component(float x) {
    return isfinite(x) ? x : 0.0f;
}

__device__ inline color sanitize(color c) {
    return color(safe_component(c.x()), safe_component(c.y()), safe_component(c.z()));
}

__device__ color ray_color(const ray& r, const sphere* spheres, int sphere_count, int depth, RNG& rng) {
    // If we've exceeded the ray bounce limit, approximate remaining contribution by background
    if (depth <= 0) {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
    }

    hit_record temp_rec;
    bool hit_anything = false;
    float closest = 1e30f;

    // brute-force hit over spheres
    for (int i=0; i<sphere_count; ++i) {
        hit_record rec;
        if (spheres[i].hit(r, 0.001f, closest, rec)) {
            hit_anything = true;
            closest = rec.t;
            temp_rec = rec;
        }
    }

    if (hit_anything) {
        ray scattered;
        color attenuation;
        bool ok = false;
        if (temp_rec.material_type == 0) {
            ok = lambertian_scatter(r, temp_rec, rng, attenuation, scattered);
        } else if (temp_rec.material_type == 1) {
            ok = metal_scatter(r, temp_rec, rng, attenuation, scattered);
        } else {
            ok = dielectric_scatter(r, temp_rec, rng, attenuation, scattered);
        }
        if (ok) return sanitize(attenuation * ray_color(scattered, spheres, sphere_count, depth-1, rng));
        // Fallback to background when a material absorption would otherwise yield black
        vec3 unit_direction_bg = unit_vector(r.direction());
        float t_bg = 0.5f*(unit_direction_bg.y() + 1.0f);
        return sanitize((1.0f-t_bg)*color(1.0f, 1.0f, 1.0f) + t_bg*color(0.5f, 0.7f, 1.0f));
    }

    // background gradient
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return sanitize((1.0f-t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f));
}

__global__ void render_kernel(color* framebuffer, int image_width, int image_height, int samples_per_pixel, int max_depth,
                              CameraParams cam, const sphere* spheres, int sphere_count, int y0, int y1) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;
    int y = y0 + y_local;
    
    if (x >= image_width || y < y0 || y >= y1) return;
    int idx = y * image_width + x;

    RNG rng( (uint32_t)(idx*9781u + 1u) );

    color pixel_color(0,0,0);
    #if DEBUG_FLAT
        // Simple debug gradient to verify kernel launch and framebuffer writes
        float u = (float)x / fmaxf(1.0f, (float)(image_width - 1));
        float v = (float)y / fmaxf(1.0f, (float)(image_height - 1));
        pixel_color = color(u, v, 0.2f);
    #elif DEBUG_BG
        // Camera-based background gradient without scene intersections
        float u = (x + 0.5f) / fmaxf(1.0f, (float)(image_width - 1));
        float v = (y + 0.5f) / fmaxf(1.0f, (float)(image_height - 1));
        ray r = ray(cam.origin, cam.lower_left_corner + u*cam.horizontal + v*cam.vertical - cam.origin);
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        pixel_color = (1.0f-t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
    #else
        for (int s=0; s<samples_per_pixel; ++s) {
            float u = (x + random_float(rng)) / (image_width - 1);
            float v = (y + random_float(rng)) / (image_height - 1);
            ray r = ray(cam.origin, cam.lower_left_corner + u*cam.horizontal + v*cam.vertical - cam.origin);
            color ray_col = ray_color(r, spheres, sphere_count, max_depth, rng);
            // Sanitize each sample to avoid NaN poisoning the accumulation
            ray_col = sanitize(ray_col);
            pixel_color += ray_col;
        }
    #endif
    
    framebuffer[idx] = sanitize(pixel_color);
}

extern "C" void launch_render(color* d_fb, int image_width, int image_height, int samples_per_pixel, int max_depth,
                               const CameraParams& cam, const sphere* d_spheres, int sphere_count, int y0, int y1) {
    dim3 block(16,16);
    int rows = y1 - y0;
    dim3 grid((image_width+block.x-1)/block.x, (rows+block.y-1)/block.y);
    render_kernel<<<grid, block>>>(d_fb, image_width, image_height, samples_per_pixel, max_depth, cam, d_spheres, sphere_count, y0, y1);
    cudaDeviceSynchronize();
}
