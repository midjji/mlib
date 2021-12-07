
#pragma once
#include <mlib/utils/cvl/polynomial.h>
#include <iostream>


namespace cvl{

double cardinal_basis(double time, int degree, int derivative);
double cumulative_cardinal_basis(double time, int degree, int derivative);
double forward_extrapolation_cumulative_cardinal_basis(double time, int degree, int derivative);
double backward_extrapolation_cumulative_cardinal_basis(double time, int degree, int derivative);


template<class T>
T basis_spline(int degree, int i, T t, T kndist, int derivative)
{
    if(degree<derivative) return T(0);
    assert(degree>=0);
    T x= t/kndist -T(i);
    T v=cardinal_basis(x,degree,derivative);
    if(derivative!=0) v*=std::pow(1.0/kndist,double(derivative));
    return v;
}

template<class T> double
cumulative_basis_spline(int degree,
                        int i,
                        T t,
                        double delta_time,
                        int derivative){


    double x=t/delta_time -double(i);
    double v=0;
    // delta_time must be 1 for this to work, so move derivative factor out of it

    for(int j=0;j<degree+2;++j) v+=basis_spline(degree,j,x,1.0,derivative);




    if(std::abs(v)<-1e-12)  v=0;
    // max of cum basis is 1.0, improves num
    if(derivative==0) return (std::abs(v-1.0) < 1e-12) ? v: 1.0;
    v*=std::pow(1.0/delta_time,double(derivative));
    return v;
}



template<int degree, class Type=long double>
/**
 * @brief get_spline_basis_polys
 * @return
 *
 * This is the basis polynomial
 *
 * it is tested!
 */
CompoundBoundedPolynomial<degree,Type> get_spline_basis_polys() {
    CompoundBoundedPolynomial<degree,Type> ret;
    if constexpr (degree==0){
        ret.add(BoundedPolynomial<0,Type>(Vector2<long double>(0,1),Polynomial<0,Type>(1)));

    }
    else
    {
        long double k=1.0/((long double)degree);
        CompoundBoundedPolynomial<degree,Type> a=
                get_spline_basis_polys<degree-1,Type>()*
                (Polynomial<1,Type>(0,1)*Type(k));

        CompoundBoundedPolynomial<degree,Type> b=
                get_spline_basis_polys<degree-1,Type>().reparam(-1)*
                (Polynomial<1,Type>(degree+1,-1)*Type(k));

        for(auto bp:a.polys)ret.add(bp);
        for(auto bp:b.polys)ret.add(bp);

    }
    ret.collapse();
    return ret;
}

double get_spline_basis_polys_integer_factor(int degree);




template<int degree>

CompoundBoundedPolynomial<degree>
get_spline_cumulative_basis_polys()
{ // this
    CompoundBoundedPolynomial<degree> ret;
    for(int j=0;j<degree+1 ;++j) {
        auto b=get_spline_basis_polys<degree>().reparam(-j);
        for(auto p:b.polys) ret.add(p);
    }
    ret.collapse();
    ret.bound(Vector2<long double>(0,degree));
    ret.add(BoundedPolynomial<degree>({degree,std::numeric_limits<long double>::max()},1));
    ret.collapse();
    return ret;
}

template<int degree>
CompoundBoundedPolynomial<degree>
forward_cumulative_extrapolation_basis()
{ // this is for the cumulative cardinal basis
    // tis is offset by L which is the index of the last interior control point.
    CompoundBoundedPolynomial<degree> ret;
    if constexpr(degree==2){

        ret.add(BoundedPolynomial<degree>(
                    {0,1},0,0,0.5));
        // then its x forever...
        ret.add(BoundedPolynomial<degree>(
                    {1,std::numeric_limits<long double>::max()},-0.5,1));
    }

    if constexpr (degree==3){
        //bounds: [0,1) p(x) = 1x^3 /6
        ret.add(BoundedPolynomial<degree>(
                    {0,1},0,0,0,1/6.0));
        //bounds: [1,2) p(x) = (-2x^3 + 9x^2 + -9x + 3)/6

        ret.add(BoundedPolynomial<degree>(
                    {1,2},2/6.0,-1,1,-1/6.0));
        // then its x forever...
        ret.add(BoundedPolynomial<degree>(
                    {2,std::numeric_limits<long double>::max()},-1,1));
    }
    if constexpr (degree==4){

        ret.add(BoundedPolynomial<degree>(
                    {0,1},0,0,0,0,1/24.0));

        //bounds: [1,2) p(x) = -2x^4 + 12x^3 + -18x^2 + 12x + -3 )/24
        ret.add(BoundedPolynomial<degree>(
                    {1,2},-3/24.0, 12/24.0,-18/24.0, 12/24.0,-2/24.0));
        //bounds: [2,3) p(x) = 1x^4 + -12x^3 + 54x^2 + -84x + 45/ 24
        ret.add(BoundedPolynomial<degree>(
                    {2,3},45/24.0, -84/24.0,54/24.0, -12/24.0,1/24.0));
        // then its x forever...
        ret.add(BoundedPolynomial<degree>(
                    {3,std::numeric_limits<long double>::max()},-1.5,1));
    }
    if constexpr (degree==5){
        ret.add(BoundedPolynomial<degree>(
                    {0,1},0,0,0,0,0,1/120.0));

        ret.add(BoundedPolynomial<degree>(
                    {1,2},4.0/120.0,-20.0/120.0, 40.0/120.0,-40.0/120.0, 20.0/120.0,-3.0/120.0));

        ret.add(BoundedPolynomial<degree>(
                    {2,3},-188.0/120.0,460.0/120.0, -440.0/120.0,200.0/120.0, -40.0/120.0,3.0/120.0));
        ret.add(BoundedPolynomial<degree>(
                    {3,4},784.0/120.0, -1160.0/120.0,640.0/120.0, -160.0/120.0,20.0/120.0,-1.0/120.0));
        // then its x forever...
        ret.add(BoundedPolynomial<degree>(
                    {4,std::numeric_limits<long double>::max()},-2,1));
    }
    ret.collapse();
    return ret;
}
template<int degree>
CompoundBoundedPolynomial<degree>
backwards_cumulative_extrapolation_basis()
{ // this is for the cumulative cardinal basis
    // tis is offset by L which is the index of the last interior control point.
    CompoundBoundedPolynomial<degree> ret;
    if(degree==2){

        // then its x forever...
        ret.add(BoundedPolynomial<degree>(
                    {-std::numeric_limits<long double>::max(),1},-0.5,1));

        ret.add(BoundedPolynomial<degree>(
                    {1,2},-1,2,-0.5));

        // then its x forever...
        ret.add(BoundedPolynomial<degree>(
                    {2,std::numeric_limits<long double>::max()},1,0));
    }

    if constexpr (degree==3){


        ret.add(BoundedPolynomial<degree>(
                    {1,2},-6 + 1.0,3,3,-1));
        ret.add(BoundedPolynomial<degree>(
                    {2,3},-21,27,-9,1));

        ret*=1.0/6.0;
        ret.add(BoundedPolynomial<degree>({-std::numeric_limits<long double>::max(),1}, -1,1));
        ret.add(BoundedPolynomial<degree>({3,std::numeric_limits<long double>::max()},1,0));


    }
    if constexpr (degree==4){

        ret.add(BoundedPolynomial<degree>(
                    {1,2},11 -2*24 ,28,-6,4,-1));
        ret.add(BoundedPolynomial<degree>(
                    {2,3},35-24,-68,66,-20,2));
        ret.add(BoundedPolynomial<degree>(
                    {3,4},-232,256,-96,16,-1));
        ret*=1.0/24.0;
        ret.add(BoundedPolynomial<degree>({-std::numeric_limits<long double>::max(),1}, -1.5,1));
        ret.add(BoundedPolynomial<degree>({4,std::numeric_limits<long double>::max()},1,0));


    }
    if constexpr (degree==5){

        ret.add(BoundedPolynomial<degree>(
                    {1,2},121-3*120,115,10,-10,5,-1));
        ret.add(BoundedPolynomial<degree>(
                    {2,3},-127-2*120 ,435,-310,150,-35,3));
        ret.add(BoundedPolynomial<degree>(
                    {3,4},1211-120,-1995,1310,-390,55,-3));
        ret.add(BoundedPolynomial<degree>(
                    {4,5},-3005,3125,-1250,250,-25,1));

        ret*=1.0/120.0;
        ret.add(BoundedPolynomial<degree>({-std::numeric_limits<long double>::max(),1}, -2,1));
        ret.add(BoundedPolynomial<degree>({5,std::numeric_limits<long double>::max()},1,0));
    }
    ret.collapse();
   return ret.reparam(-1);
    //return ret;
}
}// end namespace cvl

