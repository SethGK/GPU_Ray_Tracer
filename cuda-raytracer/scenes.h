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
// Scene: Packed spheres in a uniform 3D grid (touching, no overlap)
hittable_list packed_spheres() {
    hittable_list world;

    // Easy-to-tweak parameters
    const int Nx = 6;   // number along X
    const int Ny = 3;   // number along Y
    const int Nz = 6;   // number along Z
    const float radius = 0.4f;
    const float spacing = 2.0f * radius; // touching

    // Optional ground
    world.add({ point3(0,-1000.0f,0), 1000.0f, 0, vec3(0.5f,0.5f,0.5f), 0.0f, 1.0f });

    // Offset so the cluster is centered-ish around the origin
    const float ox = -0.5f * (Nx - 1) * spacing;
    const float oy = radius + 0.01f; // slightly above ground
    const float oz = -0.5f * (Nz - 1) * spacing;

    RNG rng(2025u);

    for (int iy = 0; iy < Ny; ++iy) {
        for (int ix = 0; ix < Nx; ++ix) {
            for (int iz = 0; iz < Nz; ++iz) {
                point3 c(ox + ix * spacing, oy + iy * spacing, oz + iz * spacing);

                // Randomize material: 0=diffuse, 1=metal, 2=dielectric
                int m = (int)floorf(random_float(rng, 0.0f, 3.0f));
                vec3 albedo(0.7f, 0.7f, 0.7f);
                float fuzz = 0.0f;
                float ir = 1.0f;
                if (m == 0) {
                    // Diffuse with random color
                    albedo = vec3(random_float(rng, 0.2f, 0.9f), random_float(rng, 0.2f, 0.9f), random_float(rng, 0.2f, 0.9f));
                } else if (m == 1) {
                    // Metal with slight fuzz
                    albedo = vec3(random_float(rng, 0.6f, 0.95f), random_float(rng, 0.6f, 0.95f), random_float(rng, 0.6f, 0.95f));
                    fuzz = random_float(rng, 0.0f, 0.25f);
                } else {
                    // Dielectric (glass)
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
