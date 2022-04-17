#pragma once
#include "ray.h"

template<class T> class Material;

template<class T>
struct hit_record
{
    T t;
    Vector3<T> p;
    Vector3<T> normal;
    Material<T> mat;
};


