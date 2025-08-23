#ifndef COLOR_H
#define COLOR_H

#include <cstdio>
#include "vec3.h"

inline void write_color(FILE* f, color pixel_color, int samples_per_pixel) {
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    // Divide by samples and gamma-correct with gamma=2.0
    float scale = 1.0f / samples_per_pixel;
    r = sqrtf(fmaxf(0.0f, r * scale));
    g = sqrtf(fmaxf(0.0f, g * scale));
    b = sqrtf(fmaxf(0.0f, b * scale));

    int ir = static_cast<int>(256.0f * fminf(fmaxf(r, 0.0f), 0.999f));
    int ig = static_cast<int>(256.0f * fminf(fmaxf(g, 0.0f), 0.999f));
    int ib = static_cast<int>(256.0f * fminf(fmaxf(b, 0.0f), 0.999f));

    std::fprintf(f, "%d %d %d\n", ir, ig, ib);
}

#endif
