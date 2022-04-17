#pragma once




#include "hitable.h"
#include "ray.h"

#include <mlib/utils/random.h>
#include <mlib/utils/cvl/matrix.h>


using namespace cvl;





template<class T>
  __host__ __device__
inline T schlick(T cosine, T ref_idx)
{
    T r0 = (T(1)-ref_idx) / (T(1)+ref_idx);
    r0 = r0*r0;
    return r0 + (T(1)-r0)*pow((T(1) - cosine),5);
}

template<class T>
  __host__ __device__
inline bool refract(const Vector3<T>& v, const Vector3<T>& n, T ni_over_nt, Vector3<T>& refracted)
{
    Vector3<T> uv = v/v.length();
    T dt = dot(uv, n);
    T discriminant = T(1.0) - ni_over_nt*ni_over_nt*(T(1)-dt*dt);
    if (discriminant > T(0)) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

template<class T>
  __host__ __device__
inline Vector3<T> reflect(const Vector3<T>& v, const Vector3<T>& n) {
    return v - T(2.0)*dot(v,n)*n;
}

template<class T, class Random>
  __host__ __device__
inline Vector3<T> random_in_unit_sphere(Random& random)
{
    Vector3<T> p;
    do {
        p = Vector3<T>(random.u(-1,1),random.u(-1,1),random.u(-1,1));
    } while (p.squaredNorm() >= T(1.0));
    return p;
}

template<class T, class Random>
           __host__ __device__
bool lambertian_scatter(const Ray<T>& r_in, const hit_record<T>& rec, Vector3<T>& attenuation, Ray<T>& scattered, const Vector3<T>&  albedo, Random& random)
{
    Vector3<T> target = rec.normal + random_in_unit_sphere<T>(random);
    scattered = Ray<T>(rec.p, target );
    attenuation = albedo;
    return true;
}

template<class T, class Random>
           __host__ __device__
bool metal_scatter(const Ray<T>& r_in, const hit_record<T>& rec, Vector3<T>& attenuation, Ray<T>& scattered, const Vector3<T>& albedo, T fuzz, Random& random)
{
    Vector3<T> reflected = reflect(r_in.direction.normalized(), rec.normal);
    scattered = Ray<T>(rec.p, (reflected + fuzz*random_in_unit_sphere<T>(random)));
    attenuation = albedo;
    return (dot(scattered.direction, rec.normal) > T(0));
}



template<class T, class Random>
           __host__ __device__
bool dielectric_scatter(const Ray<T>& r_in, const hit_record<T>& rec, Vector3<T>& attenuation, Ray<T>& scattered, T ref_idx, Random& random)  {
    Vector3<T> outward_normal;
    Vector3<T> reflected = reflect(r_in.direction, rec.normal);
    T ni_over_nt;
    attenuation = Vector3<T>(1.0, 1.0, 1.0);
    Vector3<T> refracted;
    T reflect_prob;
    T cosine;
    if (dot(r_in.direction, rec.normal) > T(0))
    {
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
        //         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        cosine = dot(r_in.direction, rec.normal) / r_in.direction.length();
        cosine = sqrt(T(1) - ref_idx*ref_idx*(T(1)-cosine*cosine));
    }
    else
    {
        outward_normal = rec.normal;
        ni_over_nt = T(1.0) / ref_idx;
        cosine = -dot(r_in.direction, rec.normal) / r_in.direction.length();
    }
    if (refract(r_in.direction, outward_normal, ni_over_nt, refracted))
        reflect_prob = schlick(cosine, ref_idx);
    else
        reflect_prob = T(0);
    if (random.u(0,1) < reflect_prob)
        scattered = Ray<T>(rec.p, reflected);
    else
        scattered = Ray<T>(rec.p, refracted);
    return true;
}



template<class T>
class Material
{
    int material;
    Vector3<T> albedo;
    T fuzz;
    T ref_idx;

public:
        __host__ __device__
    Material()=default;
            __host__ __device__
    Material(Vector3<T> albedo):material(0), albedo(albedo){}
                __host__ __device__
    Material(Vector3<T> albedo, T fuzz):material(1), albedo(albedo), fuzz(fuzz){
                    if (fuzz > 1) fuzz = T(1.0);
                }
                    __host__ __device__
    Material(T ref_idx):material(2), ref_idx(ref_idx){}
                               template<class Random>
           __host__ __device__

    bool scatter(const Ray<T>& r_in, const hit_record<T>& rec, Vector3<T>& attenuation, Ray<T>& scattered, Random& random ) const{

        switch(material)
        {
        case 0: return lambertian_scatter(r_in, rec, attenuation, scattered, albedo, random);
        case 1: return metal_scatter(r_in, rec, attenuation, scattered, albedo, fuzz, random);
        case 2: return dielectric_scatter(r_in, rec, attenuation, scattered, ref_idx, random);

        }
    }

};



