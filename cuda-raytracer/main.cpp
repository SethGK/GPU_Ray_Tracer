#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "vec3.h"
#include "color.h"
#include "camera.h"
#include "sphere.h"

// Host wrapper provided by kernels.cu
extern "C" void launch_render(color* d_fb, int image_width, int image_height, int samples_per_pixel, int max_depth,
                               const camera& cam, const sphere* d_spheres, int sphere_count, int y0, int y1);

int main() {
    // Image
    const int image_width = 800;
    const int image_height = 600;
    const int samples_per_pixel = 50;
    const int max_depth = 10;

    // Camera
    camera cam(point3(3,3,2), point3(0,0,-1), vec3(0,1,0), 20.0f, float(image_width)/image_height);

    // Scene: ground + three spheres
    std::vector<sphere> host_spheres;
    // ground
    host_spheres.push_back({ point3(0,-100.5f,-1), 100.0f, 0, vec3(0.8f,0.8f,0.0f), 0.0f, 1.0f });
    // center diffuse
    host_spheres.push_back({ point3(0,0,-1), 0.5f, 0, vec3(0.7f,0.3f,0.3f), 0.0f, 1.0f });
    // left glass
    host_spheres.push_back({ point3(-1,0,-1), 0.5f, 2, vec3(1.0f,1.0f,1.0f), 0.0f, 1.5f });
    // right metal
    host_spheres.push_back({ point3(1,0,-1), 0.5f, 1, vec3(0.8f,0.6f,0.2f), 0.0f, 1.0f });

    sphere* d_spheres = nullptr;
    cudaMalloc(&d_spheres, host_spheres.size()*sizeof(sphere));
    cudaMemcpy(d_spheres, host_spheres.data(), host_spheres.size()*sizeof(sphere), cudaMemcpyHostToDevice);

    // Framebuffer
    color* d_fb = nullptr;
    cudaMalloc(&d_fb, image_width*image_height*sizeof(color));

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
