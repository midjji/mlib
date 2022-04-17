#pragma once
#include <mlib/utils/cvl/matrix.h>
using namespace cvl;

template<class T>
class Ray
{
    public:
            __host__ __device__
        Ray()=default;
                __host__ __device__
        Ray(const Vector3<T>& origin, const Vector3<T>& direction):origin(origin),direction(direction)
        {
            //if(std::abs(direction.squaredNorm()-1.0)>1e-10) std::cout<<direction.squaredNorm()<<"WTF"<<std::endl;
        }
                        __host__ __device__
        inline Vector3<T> operator()(T t) const { return origin + t*direction; }

        Vector3<T> origin;
        Vector3<T> direction;
};
