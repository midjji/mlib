#pragma once
#include <mlib/utils/cvl/matrix.h>
namespace cvl{
double cardinal_basis(double time, int degree, int derivative);
double cumulative_cardinal_basis(double time, int degree, int derivative);
double forward_extrapolation_cumulative_cardinal_basis(double time, int degree, int derivative);
double backward_extrapolation_cumulative_cardinal_basis(double time, int degree, int derivative);

// Multidim
double cardinal_basis(double x, double y, int degree, int derivative);



// interface for coefficients

class SplineBasis
{
    double delta_time;
    int degree;
    int first_control_point;
    int last_control_point;

public:
    SplineBasis()=default;
    SplineBasis(double delta_time, int degree, int first_control_point, int last_control_point):
        delta_time(delta_time),degree(degree), first_control_point(first_control_point), last_control_point(last_control_point){}
    double operator()(double time, int index, int derivative) const;
    int get_first(double time) const;
    int get_last(double time) const;
};

class SplineBasisKoeffs
{
    double time;
    SplineBasis sb;
public:
    SplineBasisKoeffs(double time, SplineBasis sb):time(time), sb(sb){};
    double operator()(int index, int derivative=0) const
    {
        return sb(time, index, derivative);
    }
    template<int Degree> inline Vector<double,Degree+1> cbs(int derivative=0) const
    {
        Vector<double,Degree+1> ret;
        auto& a=*this;
        int j=0;
        for(int i=sb.get_first(time);i<=sb.get_last(time);++i){
            ret[j]=a(i,derivative);
            j++;
        }
        return ret;
    }
    template<int Degree, int derivatives=3> inline
    Vector<Vector<double,Degree+1>,derivatives>
    cbss() const
    {
        Vector<Vector<double,Degree+1>,derivatives> ret;
        for(int i=0;i<derivatives;++i)
            ret[i]=cbs<Degree>(i);
        return ret;
    }
};

}// end namespace cvl
