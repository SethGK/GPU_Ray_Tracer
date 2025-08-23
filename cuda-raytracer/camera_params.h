#ifndef CAMERA_PARAMS_H
#define CAMERA_PARAMS_H

#include "vec3.h"

struct CameraParams {
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};

#endif
