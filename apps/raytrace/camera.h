#ifndef CAMERAH
#define CAMERAH
#include "ray.h"

template<class T> Vector3<T> random_in_unit_disk() {
    Vector3<T> p;
    do {
        p = T(2.0)*Vector3<T>(drand48(),drand48(),0) - Vector3<T>(1,1,0);
    } while (dot(p,p) >= 1.0);
    return p;
}

template<class T>
class Camera {
    public:
            __host__ __device__
    Camera()=default;
                    __host__ __device__
        Camera(Vector3<T> lookfrom, Vector3<T> lookat, Vector3<T> vup, T vfov, T aspect, T aperture, T focus_dist)
        { // vfov is top to bottom in degrees
            lens_radius = aperture / T(2);
            T theta = vfov*3.1415/T(180);
            T half_height = tan(theta/2);
            T half_width = aspect * half_height;
            origin = lookfrom;
            w = (lookfrom - lookat).normalized();
            u = cross(vup, w).normalized();
            v = cross(w, u);
            lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
            horizontal = T(2)*half_width*focus_dist*u;
            vertical = T(2)*half_height*focus_dist*v;
        }
        __host__ __device__
        inline Ray<T> get_ray(T s, T t) const
        {
            return Ray<T>(origin, (lower_left_corner + s*horizontal + t*vertical - origin));
        }

        Vector3<T> origin;
        Vector3<T> lower_left_corner;
        Vector3<T> horizontal;
        Vector3<T> vertical;
        Vector3<T> u, v, w;
        T lens_radius;
};
#endif




