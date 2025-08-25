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
    const int samples_per_pixel = 1000;
    const int max_depth = 50;

    // Camera - positioned to see the packed spheres scene better
    point3 lookfrom(8.0f, 6.0f, 8.0f);  // Higher and further back
    point3 lookat(0.0f, 3.0f, 0.0f);    // Looking at the center of the sphere cluster
    vec3 vup(0,1,0);
    float vfov = 30.0f;  // Slightly narrower field of view
    camera cam_host(lookfrom, lookat, vup, vfov, float(image_width)/image_height);
    CameraParams cam{};
    // Recompute camera params to pass as POD
    {
        float theta = degrees_to_radians(vfov);
        float h = tanf(theta/2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = (float)image_width/(float)image_height * viewport_height;
        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);
        cam.origin = lookfrom;
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
        // Default scene logic updated for more realism
        RNG rng(1337u);
        // Ground plane
        world.add({point3(0,-1000.0f,0), 1000.0f, 1, vec3(0.8f,0.8f,0.8f), 0.1f, 1.0f});

        // Emissive light source
        world.add({point3(0, 4, 0), 1.5f, 3, vec3(4.0f, 4.0f, 4.0f), 0.0f, 0.0f});

        // Central spheres
        world.add({point3(0, 1, 0), 1.0f, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f}); // Dielectric
        world.add({point3(-2, 1, -1), 1.0f, 1, vec3(0.7f, 0.6f, 0.5f), 0.05f, 1.0f}); // Metal
        world.add({point3(2, 1, 1), 1.0f, 0, vec3(0.4f, 0.2f, 0.1f), 0.0f, 1.0f}); // Diffuse

        // Random smaller spheres
        const int num_spheres = 200;
        for (int i = 0; i < num_spheres; ++i) {
            point3 center(random_float(rng, -10.0f, 10.0f), 0.2f, random_float(rng, -10.0f, 10.0f));
            if ((center - point3(4,0.2,0)).length() > 0.9) {
                float sphere_radius = 0.2f;
                int material_type = (int)floorf(random_float(rng, 0.0f, 3.0f));
                vec3 albedo;
                float fuzz = 0.0f, ref_idx = 1.5f;
                if (material_type == 0) { // Diffuse
                    albedo = vec3(random_float(rng, 0.0f, 1.0f), random_float(rng, 0.0f, 1.0f), random_float(rng, 0.0f, 1.0f));
                } else if (material_type == 1) { // Metal
                    albedo = vec3(random_float(rng, 0.5f, 1.0f), random_float(rng, 0.5f, 1.0f), random_float(rng, 0.5f, 1.0f));
                    fuzz = random_float(rng, 0.0f, 0.2f);
                } else { // Dielectric
                    albedo = vec3(1.0f, 1.0f, 1.0f);
                    ref_idx = 1.5f;
                }
                world.add({center, sphere_radius, material_type, albedo, fuzz, ref_idx});
            }
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
