#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include "vec3.h"
#include "utils.h"
#include "color.h"
#include "camera.h"
#include "camera_params.h"
#include "sphere.h"
#include "scenes.h"

// Host wrapper provided by kernels.cu
extern "C" void launch_render(color* d_fb, int image_width, int image_height, int samples_per_pixel, int max_depth,
                               const CameraParams& cam, const sphere* d_spheres, int sphere_count, int y0, int y1);

int main(int argc, char* argv[]) {
    // Image (1920x1080 for desktop background)
    const int image_width = 1920;
    const int image_height = 1080;
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

    hittable_list world;
    std::string scene_name = "default";

    if (argc > 2 && std::string(argv[1]) == "--scene") {
        scene_name = argv[2];
    }

    if (scene_name == "crystal") {
        std::cout << "Rendering scene: Crystal Cluster" << std::endl;
        world = crystal_cluster();
    } else if (scene_name == "molecule") {
        std::cout << "Rendering scene: Molecule" << std::endl;
        world = molecule();
    } else if (scene_name == "recursive") {
        std::cout << "Rendering scene: Recursive Sculpture" << std::endl;
        world = recursive_sculpture();
    } else if (scene_name == "packed") {
        std::cout << "Rendering scene: Packed Spheres" << std::endl;
        world = packed_spheres();
    } else {
        std::cout << "Rendering scene: Default Spiral" << std::endl;
        // Default scene logic from before
        RNG rng(1337u);
        world.add({point3(0,-1000.0f,0), 1000.0f, 0, vec3(0.5f,0.5f,0.5f), 0.0f, 1.0f});
        world.add({point3(0, 1, 0), 1.0f, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f});
        world.add({point3(-4, 1, 0), 1.0f, 0, vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f});
        world.add({point3(4, 1, 0), 1.0f, 1, vec3(0.7f, 0.6f, 0.5f), 0.1f, 1.0f});
        const int num_spheres_in_spiral = 150;
        for (int i = 0; i < num_spheres_in_spiral; ++i) {
            float fraction = (float)i / (num_spheres_in_spiral - 1);
            float angle = fraction * 4 * 2.0f * M_PI;
            float radius = 2.0f + fraction * 6.0f;
            float y = 0.2f + fraction * 10.0f;
            point3 center(radius*cosf(angle), y, radius*sinf(angle));
            float sphere_radius = 0.2f + fraction*0.3f;
            int material_type = i % 3;
            vec3 albedo;
            float fuzz = 0.0f, ref_idx = 1.0f;
            if (material_type == 0) albedo = vec3(0.1f + fraction*0.8f, 0.5f, 1.0f - fraction);
            else if (material_type == 1) { albedo = vec3(0.8f,0.8f,0.8f); fuzz = fraction*0.3f; }
            else { albedo = vec3(1.0f,1.0f,1.0f); ref_idx = 1.3f + fraction*0.4f; }
            world.add({center, sphere_radius, material_type, albedo, fuzz, ref_idx});
        }
    }

    std::vector<sphere> host_spheres = world.objects;

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
