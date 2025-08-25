#ifndef SCENES_H
#define SCENES_H

#include <vector>
#include "sphere.h"
#include "hittable_list.h"
#include "utils.h"

// Scene: A cluster of dielectric (glass) spheres, with some emissive ones inside.
hittable_list crystal_cluster() {
    hittable_list world;
    RNG rng(1337u);

    world.add({point3(0,-1000.5f, -1.0f), 1000.0f, 0, vec3(0.2f, 0.2f, 0.2f), 0.0f, 1.0f}); // Ground

    // Central emissive sphere (hidden)
    world.add({point3(0.0f, 0.5f, -1.0f), 0.3f, 3, vec3(4.0f, 2.5f, 1.0f), 0.0f, 1.0f});

    for (int i = 0; i < 15; ++i) {
        point3 center(
            random_float(rng, -1.5f, 1.5f),
            random_float(rng, 0.2f, 2.0f),
            -1.0f + random_float(rng, -1.5f, 1.5f)
        );
        float radius = random_float(rng, 0.2f, 0.5f);
        world.add({center, radius, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f}); // Dielectric
    }
    return world;
}

// Scene: A simple molecule structure (like methane).
hittable_list molecule() {
    hittable_list world;
    world.add({point3(0,-1000.0f,0), 1000.0f, 0, vec3(0.5f,0.5f,0.5f), 0.0f, 1.0f}); // Ground

    // Carbon atom (central diffuse sphere)
    world.add({point3(0.0f, 1.0f, 0.0f), 0.5f, 0, vec3(0.1f, 0.1f, 0.1f), 0.0f, 1.0f});

    // Hydrogen atoms (smaller dielectric spheres)
    float h_radius = 0.25f;
    float bond_len = 0.8f;
    world.add({point3(0, 1.0f + bond_len, 0), h_radius, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f});
    world.add({point3(bond_len * 0.9428f, 1.0f - bond_len * 0.3333f, 0), h_radius, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f});
    world.add({point3(-bond_len * 0.4714f, 1.0f - bond_len * 0.3333f, bond_len * 0.8165f), h_radius, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f});
    world.add({point3(-bond_len * 0.4714f, 1.0f - bond_len * 0.3333f, -bond_len * 0.8165f), h_radius, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f});

    // Add a metal sphere for reflections
    world.add({point3(-2.0f, 0.8f, -1.0f), 0.8f, 1, vec3(0.8f, 0.6f, 0.2f), 0.1f, 1.0f});

    return world;
}

// Scene: A recursive sculpture with nested spheres.
hittable_list recursive_sculpture() {
    hittable_list world;
    world.add({point3(0,-1000.0f,0), 1000.0f, 0, vec3(0.5f,0.5f,0.5f), 0.0f, 1.0f}); // Ground

    // Large outer sphere
    world.add({point3(0.0f, 1.5f, 0.0f), 1.5f, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f});

    // Medium sphere inside
    world.add({point3(0.0f, 1.5f, 0.0f), 0.8f, 2, vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f});

    // Small metal sphere at the core
    world.add({point3(0.0f, 1.5f, 0.0f), 0.3f, 1, vec3(0.8f, 0.6f, 0.2f), 0.0f, 1.0f});

    return world;
}


#include <cmath>
// Scene: Packed spheres filling the viewport
hittable_list packed_spheres() {
    hittable_list world;

    // Increased number of spheres for better coverage
    const int Nx = 12;   // number along X
    const int Ny = 8;    // number along Y
    const int Nz = 12;   // number along Z
    const float radius = 0.3f;  // Slightly smaller radius to fit more spheres
    const float spacing = 2.1f * radius; // Slightly overlapping for denser packing

    // Center the sphere cluster in the scene
    const float ox = -0.5f * (Nx - 1) * spacing;
    const float oy = 0.0f;  // Start from y=0
    const float oz = -0.5f * (Nz - 1) * spacing;

    RNG rng(2025u);

    // Add main cluster of spheres
    for (int iy = 0; iy < Ny; ++iy) {
        for (int ix = 0; ix < Nx; ++ix) {
            for (int iz = 0; iz < Nz; ++iz) {
                // Add some randomness to the positions for a more organic look
                float rx = random_float(rng, -0.1f, 0.1f) * spacing;
                float ry = random_float(rng, -0.1f, 0.1f) * spacing;
                float rz = random_float(rng, -0.1f, 0.1f) * spacing;
                
                point3 c(
                    ox + ix * spacing + rx,
                    oy + iy * spacing + ry,
                    oz + iz * spacing + rz
                );

                // Randomize material with different probabilities
                float mat_choice = random_float(rng);
                int m;
                vec3 albedo;
                float fuzz = 0.0f;
                float ir = 1.0f;

                if (mat_choice < 0.6f) {
                    // 60% chance for diffuse
                    m = 0;
                    albedo = vec3(
                        random_float(rng, 0.2f, 0.9f),
                        random_float(rng, 0.2f, 0.9f),
                        random_float(rng, 0.2f, 0.9f)
                    );
                } else if (mat_choice < 0.9f) {
                    // 30% chance for metal
                    m = 1;
                    albedo = vec3(
                        random_float(rng, 0.7f, 0.95f),
                        random_float(rng, 0.7f, 0.95f),
                        random_float(rng, 0.7f, 0.95f)
                    );
                    fuzz = random_float(rng, 0.0f, 0.15f);
                } else {
                    // 10% chance for dielectric
                    m = 2;
                    albedo = vec3(1.0f, 1.0f, 1.0f);
                    ir = 1.5f;
                }

                world.add({ c, radius, m, albedo, fuzz, ir });
            }
        }
    }

    return world;
}

#endif
