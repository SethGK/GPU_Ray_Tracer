#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "vec3.h"
#include "utils.h"
#include "color.h"
#include "camera.h"
#include "camera_params.h"
#include "sphere.h"

// Host wrapper provided by kernels.cu
extern "C" void launch_render(color* d_fb, int image_width, int image_height, int samples_per_pixel, int max_depth,
                               const CameraParams& cam, const sphere* d_spheres, int sphere_count, int y0, int y1);

int main() {
    // Image
    const int image_width = 800;
    const int image_height = 600;
    const int samples_per_pixel = 50;
    const int max_depth = 10; // reduced to avoid excessive device recursion depth

    // Camera
    camera cam_host(point3(13,2,3), point3(0,0,0), vec3(0,1,0), 20.0f, float(image_width)/image_height);
    CameraParams cam{};
    // Recompute camera params to pass as POD
    {
        float theta = degrees_to_radians(20.0f);
        float h = tanf(theta/2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = (float)image_width/(float)image_height * viewport_height;
        vec3 w = unit_vector(point3(13,2,3) - point3(0,0,0));
        vec3 u = unit_vector(cross(vec3(0,1,0), w));
        vec3 v = cross(w, u);
        cam.origin = point3(13,2,3);
        cam.horizontal = viewport_width * u;
        cam.vertical = viewport_height * v;
        cam.lower_left_corner = cam.origin - cam.horizontal/2.0f - cam.vertical/2.0f - w;
    }

    // Scene: large random sphere grid + big spheres
    std::vector<sphere> host_spheres;
    RNG rng(1337u);
    // ground
    host_spheres.push_back({ point3(0,-1000.0f,0), 1000.0f, 0, vec3(0.5f,0.5f,0.5f), 0.0f, 1.0f });

    for (int a = -11; a <= 11; ++a) {
        for (int b = -11; b <= 11; ++b) {
            float choose_mat = random_float(rng);
            point3 center(a + 0.9f*random_float(rng), 0.2f, b + 0.9f*random_float(rng));

            if ((center - point3(4, 0.2f, 0)).length() > 0.9f) {
                if (choose_mat < 0.8f) {
                    // diffuse
                    vec3 albedo(random_float(rng)*random_float(rng), random_float(rng)*random_float(rng), random_float(rng)*random_float(rng));
                    host_spheres.push_back({ center, 0.2f, 0, albedo, 0.0f, 1.0f });
                } else if (choose_mat < 0.95f) {
                    // metal
                    vec3 albedo(random_float(rng, 0.5f, 1.0f), random_float(rng, 0.5f, 1.0f), random_float(rng, 0.5f, 1.0f));
                    float fuzz = random_float(rng, 0.0f, 0.5f);
                    host_spheres.push_back({ center, 0.2f, 1, albedo, fuzz, 1.0f });
                } else {
                    // glass
                    host_spheres.push_back({ center, 0.2f, 2, vec3(1.0f,1.0f,1.0f), 0.0f, 1.5f });
                }
            }
        }
    }

    host_spheres.push_back({ point3(0,1,0), 1.0f, 2, vec3(1.0f,1.0f,1.0f), 0.0f, 1.5f });
    host_spheres.push_back({ point3(-4,1,0), 1.0f, 0, vec3(0.4f,0.2f,0.1f), 0.0f, 1.0f });
    host_spheres.push_back({ point3(4,1,0), 1.0f, 1, vec3(0.7f,0.6f,0.5f), 0.0f, 1.0f });

    sphere* d_spheres = nullptr;
    cudaMalloc(&d_spheres, host_spheres.size()*sizeof(sphere));
    cudaMemcpy(d_spheres, host_spheres.data(), host_spheres.size()*sizeof(sphere), cudaMemcpyHostToDevice);

    // Framebuffer
    color* d_fb = nullptr;
    cudaMalloc(&d_fb, image_width*image_height*sizeof(color));

    // Increase device stack size for recursion in device code
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);

    // Launch in row chunks and log progress
    const int chunk_rows = 32;
    for (int y0 = 0; y0 < image_height; y0 += chunk_rows) {
        int y1 = y0 + chunk_rows;
        if (y1 > image_height) y1 = image_height;
        launch_render(d_fb, image_width, image_height, samples_per_pixel, max_depth, cam, d_spheres, (int)host_spheres.size(), y0, y1);
        float pct = (100.0f * y1) / image_height;
        std::fprintf(stdout, "Rendered rows %d-%d (%.1f%%)\n", y0, y1-1, pct);
        std::fflush(stdout);
    }

    // Read back
    std::vector<color> host_fb(image_width*image_height);
    cudaMemcpy(host_fb.data(), d_fb, host_fb.size()*sizeof(color), cudaMemcpyDeviceToHost);

    // Write PPM
    FILE* f = std::fopen("output.ppm", "w");
    std::fprintf(f, "P3\n%d %d\n255\n", image_width, image_height);
    for (int j=image_height-1; j>=0; --j) {
        for (int i=0; i<image_width; ++i) {
            color c = host_fb[j*image_width + i];
            write_color(f, c, samples_per_pixel);
        }
    }
    std::fclose(f);

    // Cleanup
    cudaFree(d_fb);
    cudaFree(d_spheres);

    return 0;
}
