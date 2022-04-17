#pragma once

#include "hitable.h"
#include "material.h"

template<class T>
class Sphere
{
    public:
    __host__ __device__
    Sphere()=default;
       __host__ __device__
        Sphere(Vector3<T> cen, T r, Material<T> m) : center(cen), radius(r), mat(m)  {};
          __host__ __device__
        inline bool hit(const Ray<T>& r, T t_min, T t_max, hit_record<T>& rec) const
        {
            Vector3<T> oc = r.origin - center;
            T a = dot(r.direction, r.direction);
            T b = dot(oc, r.direction);
            T c = dot(oc, oc) - radius*radius;
            T discriminant = b*b - a*c;
            if (discriminant > 0)
            {
                T temp = (-b - sqrt(discriminant))/a;
                if (temp < t_max && temp > t_min)
                {
                    rec.t = temp;
                    rec.p = r(rec.t);
                    rec.normal = (rec.p - center) / radius;
                    rec.mat = mat;
                    return true;
                }
                temp = (-b + sqrt(discriminant)) / a;
                if (temp < t_max && temp > t_min)
                {
                    rec=hit_record<T>{temp, r(rec.t), (rec.p - center) / radius, mat};
                    return true;
                }
            }
            return false;
        }
        Vector3<T> center;
        T radius;
        Material<T> mat;
};

template<class T>
inline std::vector<Sphere<T>> random_scene(int n = 500)
{
    std::vector<Sphere<T>> spheres;spheres.reserve(n);


    spheres.emplace_back(Vector3<T>(0,-1000,0), 1000, Material<T>(Vector3<T>(0.5, 0.5, 0.5)));


    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            T choose_mat = mlib::randu_<T>(0,1);
            Vector3<T> center(a+0.9*mlib::randu_<T>(0,1),0.2,b+0.9*mlib::randu_<T>(0,1));
            if ((center-Vector3<T>(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8)
                {  // diffuse
                    spheres.emplace_back(center, 0.2, Material<T>(Vector3<T>(mlib::randu_<T>(0,1)*mlib::randu_<T>(0,1), mlib::randu_<T>(0,1)*mlib::randu_<T>(0,1), mlib::randu_<T>(0,1)*mlib::randu_<T>(0,1))));
                }
                else if (choose_mat < 0.95) { // metal
                    spheres.emplace_back(Sphere<T>(center, 0.2,
                                                   Material<T>(Vector3<T>(0.5*(1 + mlib::randu_<T>(0,1)), 0.5*(1 + mlib::randu_<T>(0,1)), 0.5*(1 + mlib::randu_<T>(0,1))),  0.5*mlib::randu_<T>(0,1))));
                }
                else
                {  // glass
                    spheres.emplace_back(Sphere<T>(center, 0.2, Material<T>(1.5)));
                }
            }
        }
    }

    spheres.emplace_back(Vector3<T>(0, 1, 0), 1.0, Material<T>(1.5));
    spheres.emplace_back(Vector3<T>(-4, 1, 0), 1.0,  Material<T>(Vector3<T>(0.4, 0.2, 0.1)));
    spheres.emplace_back(Vector3<T>(4, 1, 0), 1.0, Material<T>(Vector3<T>(0.7, 0.6, 0.5), 0.0));
    return spheres;
}

template<class T>
inline std::vector<Sphere<T>> test_scene()
{
    std::vector<Sphere<T>> spheres;
    spheres.reserve(3);
    spheres.emplace_back(Vector3<T>(0,0,-1), 0.5, Material<T>(Vector3<T>(0.1, 0.2, 0.5)));
    spheres.emplace_back(Vector3<T>(0,-100.5,-1), 100,  Material<T>(Vector3<T>(0.8, 0.8, 0.0)));
    spheres.emplace_back(Vector3<T>(1,0,-1), 0.5,  Material<T>(Vector3<T>(0.8, 0.6, 0.2), 0.0));
    spheres.emplace_back(Vector3<T>(-1,0,-1), 0.5, Material<T>(1.5));
    spheres.emplace_back(Vector3<T>(-1,0,-1), -0.45, Material<T>(1.5));
    return spheres;
}

