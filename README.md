# CUDA Ray Tracer (C++17 + CUDA)

A minimal, educational CUDA ray tracer that renders a small scene with Lambertian, Metal, and Dielectric materials using recursive ray tracing on the GPU. Builds with CMake and targets NVIDIA RTX 30-series (SM 86) using CUDA cores only.

## Features
- Pinhole camera with configurable FOV
- Materials: Lambertian (diffuse), Metal (reflective, with fuzz), Dielectric (refractive)
- Recursive ray tracing with depth control
- Gradient sky background
- One-thread-per-pixel CUDA kernel with per-row progress logging
- Output: `output.ppm`

## Build and Run
```bash
# From repo root
bash cuda-raytracer/build.sh
```
This will configure and build with CMake/NVCC and run the renderer, writing `output.ppm` in the repo root. The build script prints progress by row chunks as it renders.

## Result Image
Below is the output produced by the current configuration (PNG for GitHub preview).

<p align="center">
  <img src="output.png" alt="Ray Tracing Output (PNG)" width="600"/>
  <br/>
  <sub>Original PPM also available as <code>output.ppm</code>.</sub>
</p>

### If PPM preview does not render on your viewer
Convert PPM to PNG (requires ImageMagick):
```bash
convert output.ppm output.png
```
Then you can view it locally or embed it in Markdown:
```markdown
![Ray Tracing Output](output.png)
```

## Tuning
Edit `cuda-raytracer/main.cpp`:
- Image size: `image_width`, `image_height` (default 800x600)
- Anti-aliasing: `samples_per_pixel` (default 50)
- Recursion depth: `max_depth` (default 10)
- Camera: `camera cam(...)`
- Scene: `host_spheres` (materials: 0 = Lambertian, 1 = Metal, 2 = Dielectric)

## Project Layout
- `cuda-raytracer/` — source code and CMake project
  - `vec3.h`, `ray.h`, `utils.h`, `camera.h`, `color.h`
  - `sphere.h`, `hittable.h`, `hittable_list.h`, `material.h`
  - `kernels.cu` (CUDA kernel + host launch wrapper)
  - `main.cpp` (host setup, kernel launch, PPM output)
  - `CMakeLists.txt`, `build.sh`
- `output.ppm` — rendered image (written at repo root)

## Requirements
- NVIDIA GPU with CUDA support (Ampere/SM 86 recommended)
- CUDA Toolkit (tested with CUDA 13)
- CMake 3.18+
- A C++17 compiler

## Notes
- Only CUDA APIs are used (no RTX cores / OptiX).
- Code is annotated with `__host__ __device__` where necessary.
- Straightforward brute-force sphere intersection; suitable for learning and extension (e.g., BVH, textures, more primitives).
