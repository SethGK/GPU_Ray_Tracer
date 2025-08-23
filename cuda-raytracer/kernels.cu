#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "utils.h"
#include "camera.h"
#include "sphere.h"

__device__ color ray_color(const ray& r, const sphere* spheres, int sphere_count, int depth, RNG& rng) {
    if (depth <= 0) return color(0,0,0);

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
        if (ok) return attenuation * ray_color(scattered, spheres, sphere_count, depth-1, rng);
        return color(0,0,0);
    }

    // background gradient
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
}

__global__ void render_kernel(color* framebuffer, int image_width, int image_height, int samples_per_pixel, int max_depth,
                              camera cam, const sphere* spheres, int sphere_count, int y0, int y1) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;
    int y = y0 + y_local;
    if (x >= image_width || y < y0 || y >= y1) return;
    int idx = y * image_width + x;

    RNG rng( (uint32_t)(idx*9781u + 1u) );

    color pixel_color(0,0,0);
    for (int s=0; s<samples_per_pixel; ++s) {
        float u = (x + random_float(rng)) / (image_width - 1);
        float v = (y + random_float(rng)) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, spheres, sphere_count, max_depth, rng);
    }
    framebuffer[idx] = pixel_color;
}

extern "C" void launch_render(color* d_fb, int image_width, int image_height, int samples_per_pixel, int max_depth,
                               const camera& cam, const sphere* d_spheres, int sphere_count, int y0, int y1) {
    dim3 block(16,16);
    int rows = y1 - y0;
    dim3 grid((image_width+block.x-1)/block.x, (rows+block.y-1)/block.y);
    render_kernel<<<grid, block>>>(d_fb, image_width, image_height, samples_per_pixel, max_depth, cam, d_spheres, sphere_count, y0, y1);
    cudaDeviceSynchronize();
}
