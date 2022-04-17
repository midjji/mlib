#pragma once

#include "sphere.h"

template<class T>
class hitable_list
{
    public:
    const std::vector<Sphere<T>> spheres;
        hitable_list(std::vector<Sphere<T>> spheres):spheres(spheres) { }
        bool hit(const Ray<T>& r, float t_min, float t_max, hit_record<T>& rec) const{
            hit_record<T> temp_rec;
            bool hit_anything = false;
            double closest_so_far = t_max;
            for (const auto& sphere:spheres)
            {
                if (sphere.hit(r, t_min, closest_so_far, temp_rec))
                {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            return hit_anything;
    }
};


