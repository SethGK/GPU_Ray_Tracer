#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <vector>
#include "sphere.h"

struct hittable_list {
    std::vector<sphere> objects;
    void clear() { objects.clear(); }
    void add(const sphere& s) { objects.push_back(s); }
};

#endif
