
#include <unordered_map>
#include <vector>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/spline/coeffs.h>

namespace cvl {


int index3d(int x, int y, int z, int stride_x, int stride_y)
{
    int ind=x;
    ind*=stride_x;
    ind+=y;
    ind*=stride_y;
    ind+=z;
    return ind;
}

template<class T>
double spline3d(const T* const cpts,
                Vector3d cardinal_x,
                int degree)
{
    // cardinal x is in [0,1)
    T val=T(0.0);
    for(int x=-degree;x<1;++x)
    {
        double bx= cardinal_basis(cardinal_x[0] - x, degree,0);
        for(int y=-degree;y<1;++y)
        {
            double by= cardinal_basis(cardinal_x[1] - y, degree,0);
            for(int z=-degree;z<1;++z)
            {
                double bz= cardinal_basis(cardinal_x[2] - z, degree,0);
                int index=index3d(cardinal_x[0] -x, cardinal_x[1] -y, cardinal_x[2] -z, degree+1, degree+1);

                val+= cpts[index]*T(bx*by*bz);
            }
        }
    }
    return val;
}

template<class T>
double spline3d(std::vector<T*> cpts,
                Vector3d cardinal_x,
                int degree)
{
    // cardinal x is in [0,1)
    T val=T(0.0);
    for(int x=-degree;x<1;++x)
    {
        double bx= cardinal_basis(cardinal_x[0] - x, degree,0);
        for(int y=-degree;y<1;++y)
        {
            double by= cardinal_basis(cardinal_x[1] - y, degree,0);
            for(int z=-degree;z<1;++z)
            {
                double bz= cardinal_basis(cardinal_x[2] - z, degree,0);
                int index=index3d(cardinal_x[0] -x, cardinal_x[1] -y, cardinal_x[2] -z, degree+1, degree+1);

                val+= *cpts[index]*T(bx*by*bz);
            }
        }
    }
    return val;
}


class R3Spline
{
    using int128=long double;
    using int128p=int128*;

public:
    //References/pointers to elements remain valid in all cases, even after a rehash. except when removed, or overwritten?
    std::unordered_map<int128, double> control_points_;
    Vector3d deltas;
    int degree;

    double& cpt(Vector3i cx)
    {
        Vector4<std::int32_t> xi(0,cx[0],cx[1],cx[2]);
        int128 hv = *int128p(xi.begin());
        static_assert (sizeof(hv)==16, "must be!");
        return control_points_[hv];
    }


    std::vector<double*> cpts(Vector3d x)
    {
        std::vector<double*> rets;
        Vector3i nx = point_divide(x,deltas).floor2i();

        for(int x=nx[0] - degree;x<=nx[0];++x)
            for(int y=nx[1] - degree;y<=nx[1];++y)
                for(int z=nx[2] - degree;z<=nx[2];++z) {
                    rets.push_back(&cpt(Vector3i(x,y,z)));
                }
        return rets;
    }
    double operator()(Vector3d x)
    {
        auto cp=cpts(x);
        auto nx=point_divide(x,deltas);
        return spline3d(cp, nx - nx.floored(),degree);
    }


};
}

