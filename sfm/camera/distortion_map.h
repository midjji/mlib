#pragma once
/** \file distortion_map.h
 *
 * \brief This header contains a interpolated(taylor expansion) based undistortion lookup map using the Brown camera distortion model
 *
 * \remark
 * - c++11
 * - no dependencies
 * - tested by test_undistortion.cpp
 *
 * \Note
 *
 *
 *
 *
 * \todo
 * - template on distortion model
 * - add seriealization
 * - add automated test
 * - add internal verification
 *
 *
 * Lens Distortion:
 * Lens distortion is complicated.
 *
 * Limitations:
 * I am primarily going to consider acute-wide angle lenses and not fisheye lenses(though much of the same applies, it just gets worse).
 * I am going to assume that the light is of a single frequency.
 * I think this means that they are thin angle lenses.
 *
 * I believe real optical lenses will satisfy
 * injective, smooth, continouos, limited domain.
 * but I am not sure
 *
 * I am sure that the models need not do so however. This is an artifact of strange approximations in the derivations.
 * The optimization problem is likely not convex without additional constraints. See Ring distortion error
 *
 * From an api perspective the most general model I will consider does fullfill these though.
 * Further I will assume that distortions are a special class of injective smooth and continuous maps.
 *
 * This calls for an abstraction. Since my concern is cameras and images lets call it a camera.
 * While this high level abstraction does lose some computational opportunities, there is a huge benefit to it beeing self contained.
 *
 *
 * This map should provide
 * distorted coordinates = distort(from pinhole coordinates)
 * pinhole coordinates = undistort(distorted coordinates)
 *
 * And the helpers which allow a image to be transformed from plain to distorted and from distorted to plain
 *
 *
 * \author Mikael Persson,Andreas Robinson
 * \date 2016-10-06
 * \note MIT licence

 *
 ******************************************************************************/

// Copyright 2016
// This software is distributed under the terms of the MIT license.

#pragma once

#include <vector>
#include <fstream>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/matrix_adapter.h>
 
namespace cvl{























template <typename T>
class BrownDistortionFunction
{
public:
    T k1, k2, k3, p1, p2;

    BrownDistortionFunction(){}
    BrownDistortionFunction(const cvl::Vector5<T>& d)
    {
        k1 = (T)d(0);
        k2 = (T)d(1);
        p1 = (T)d(2);
        p2 = (T)d(3);
        k3 = (T)d(4);
    }

    /// Distort the input coordinate
    cvl::Vector<T,2> value(const cvl::Vector<T,2>& in) const
    {
        cvl::Vector<T,2> v;

        T x = in(0);
        T y = in(1);

        T x2 = x*x;
        T y2 = y*y;
        T xy = x*y;

        T r2 = x*x + y*y;
        T r4 = r2*r2;
        T r6 = r2*r2*r2;

        v(0) = x * (1 + k1*r2 + k2*r4 + k3*r6) + 2 * p1*xy + p2*(r2 + 2 * x2);
        v(1) = y * (1 + k1*r2 + k2*r4 + k3*r6) + 2 * p2*xy + p1*(r2 + 2 * y2);

        return v;
    }

    /// Get the Jacobian matrix of the distortion function at the input coordinate
    cvl::Matrix<T,2,2> jacobian(const cvl::Vector<T,2>& in) const
    {
        // Auto-differentiation would perhaps be more elegant but in this case the expressions are really simple.

        cvl::Matrix<T,2,2> J(1,0,0,1);
        //return J;


        T x = in(0);
        T y = in(1);

        T x2 = x*x;
        T y2 = y*y;
        T xy = x*y;

        T r2 = x*x + y*y;
        T r4 = r2*r2;
        T r6 = r2*r2*r2;

        T a = 1 + k1*r2 + k2*r4 + k3*r6;
        T b = 2*k1 + 4*k2*r2 + 6*k3*r4;

        J(0,0) = a + 6*p2*x + 2*p1*y + x2*b;
        J(0,1) = 2*p1*x + 2*p2*y + xy*b;
        J(1,0) = 2*p1*x + 2*p2*y + xy*b;
        J(1,1) = a + 2*p2*x + 6*p1*y + y2*b;

        return J;
    }
};


/**
 * Brown distortion model is
 *
 *
 * y= K*dist(d,x_camera.dehom);
 * yn=undist(d,dist(d,x_camera.dehom));
 *
 *
 *
 */

template<class T>
cvl::Vector2<T> distort(const cvl::Vector5<T>& d,
                        const cvl::Vector2<T>& in/* K*in=(row,col) in pixels*/){
    cvl::Vector<T,2> v;

    T x = in(0);
    T y = in(1);

    T x2 = x*x;
    T y2 = y*y;
    T xy = x*y;

    T r2 = x*x + y*y;
    T r4 = r2*r2;
    T r6 = r2*r2*r2;
    T k1, k2, k3, p1, p2;
    k1 = (T)d(0);
    k2 = (T)d(1);
    p1 = (T)d(2);
    p2 = (T)d(3);
    k3 = (T)d(4);

    v(0) = x * (T(1) + k1*r2 + k2*r4 + k3*r6) + T(2) * p1*xy + p2*(r2 + T(2) * x2);
    v(1) = y * (T(1) + k1*r2 + k2*r4 + k3*r6) + T(2) * p2*xy + p1*(r2 + T(2) * y2);


    return v;
}
template<class T> Matrix<T,3,3> invertK(Matrix<T,3,3> K){
    if(  K(1,0)==0 &&
         K(0,1)==0 &&
         K(2,0)==0 &&
         K(2,1)==0){ // strict zeros have advantages to jet etc...
        Matrix<T,3,3> Ki( 1.0/K(0,0), 0, -K(0,2)/K(0,0),
                          0, 1.0/K(1,1), -K(1,2)/K(1,1),
                          0, 0, 1.0);
        return Ki;
    }

    return Matrix<long double,3,3>(K).inverse();
}



template<class T>
/**
 * @brief The PointUndistorter class
 *
 *
 * Now In some cases its easy to undistort an image, but if the full brown camera model is used, it gets complicated....
 *
 * This is a Taylor1 interpolated lookup table with the correct coefficients
 *
 * The lookup table is y_n=L(y_dk)=L(K*y_d). So from distorted coordinate to high accuracy pin hole normalized coordinates
 *
 * The size of the lookup table can be anything, but one pixel per pixel is probably decent...
 *
 * So the lookup table must cover atleast 0-rows, 0-cols which is y_d in K^{-1}(0-rows, 0-cols)
 *
 *
 *
 * Now for each y_n find y_d
 *
 * Two kinds of lookup possible... With jacobian, or with any interpol
 *
 *
 */
class PointUndistorter
{

    template<class U> class Taylor1{
    public:
        // a: Point of approximation
        // f: Function value at a
        // J: Jacobian matrix at a
        cvl::Vector<U,2> a,f;
        cvl::Matrix<U,2,2> J;
//#warning "strange compilation bug in older gccs"
        U cost=100000000000000000;//std::numeric_limits<U>::infinity(); // why the hell does this not compile?
        int valid=false;

    };
public:


    /**
     * @brief PointUndistorter
     * @param K the cameras linear intrinsics
     * @param rows
     * @param cols
     * @param lookup_rows = 2*(rows+20) are good values or just rows+20
     * @param lookup_cols = 2*(cols+20)
     * @param d the cameras non linear intrinsics, note if there is only radial distortion there is a better way...
     */
    PointUndistorter(cvl::Matrix<T,3,3> K,
                     uint rows,
                     uint cols,
                     uint lookup_rows,
                     uint lookup_cols,
                     cvl::Vector<T,5> d):K(K),rows(rows),cols(cols),lookup_rows(lookup_rows),lookup_cols(lookup_cols){
        static_assert(T(0.1)!=T(0)," requires a real type!");
        lookup_data.resize(lookup_rows*lookup_cols);
        lookup=MatrixAdapter<Taylor1<T>>(&lookup_data[0],lookup_rows,lookup_cols);
        distortFn=BrownDistortionFunction<double>(d);
    }


    bool in(uint v, uint low, uint high){
        return ((low<=v) && (v<=(high+1)));
    }
    /**
     * @brief init
     */
    void init()
    {
        if(inited) return;

        Kinv = invertK(K);

        // compute the K of the lookup table:
        // So I want to map ykd 0,0 to say 10,10, so the
        double Lpr=0;
        double Lpc=0;
        // this allows me to leave the lookup table for a bit and still get good results...
        // I want ykd rows,cols to say lookup_rows-10,lookup_cols-10, so the
        double fr=(lookup_rows-0.0)/(double)rows;
        double fc=(lookup_cols-0.0)/(double)cols;

        L=Matrix<T,3,3> (fr,0,Lpr,
                         0,fc,Lpc,
                         0,0,1);
        Linv=invertK(L);










        std::vector<T> cost(lookup_cols * lookup_rows, std::numeric_limits<T>::infinity());




        // which yn give yd ?
        // A coarse approximation ykd approx K*yn this will work if the distortion pushes things out of the image,
        // if the distortion draws things towards the center or does both, we need to correct for that by adding another 8 passes or more...
        // this approach just wont work for fisheye, infact it can only work for


        // lets start with senario A... its pushing things outwards, if it isnt this will still provide usefull data
        for (T row = 0; row < rows; row += 0.1) {
            for (T col = 0; col < cols; col += 0.1) {

                // Get undistorted image coordinates

                Vector<T,3> yk = (Vector3<T>(row, col, 1));
                Vector<T,3> yn = Kinv*yk;
                // Get distorted coordinates and the Jacobian matrix at x2
                Vector<T,3> yd  = distortFn.value(yn.dehom()).homogeneous();
                Vector<T,3> ykd = K*yd; // where we would be in the image
                Vector<T,3> ly = L*ykd; // where we are in the lookup image


                // Get the nearest pixel grid location



                int lookup_row = (int) (ly(0)); // should be in row, col!
                int lookup_col = (int) (ly(1));
                if(lookup_col<0 || lookup_row<0) continue;

                if(lookup.is_in(lookup_row,lookup_col)){

                    // Discard the point if the distortion is worse than previously found

                    T c = (yk - ykd).norm(); // Distance between distorted and undistorted pixels
                    if (c < lookup(lookup_row,lookup_col).cost){
                        lookup(lookup_row,lookup_col).cost = c;

                        // Store the approximate inverse distortion function parameters

                        lookup(lookup_row,lookup_col).a = yd.dehom();
                        Matrix2d Jr = distortFn.jacobian(yn.dehom());
                        lookup(lookup_row,lookup_col).J = Jr.inverse();
                        lookup(lookup_row,lookup_col).f = yn.dehom();
                        lookup(lookup_row,lookup_col).valid = true;
                    }
                }
            }
        }
        // deal with invalids near valids, sample closer near them
        // use progressive refinement of the coordinates
        // deal with fast changes: sample closer near gradients in the distortion

        inited=true;
    }
BrownDistortionFunction<T> distortFn;
Vector<T,2> distort(Vector<T,2> yn){return distortFn.value(yn);}
    /**
 * Undistort a point.
 *
 * @param ykd distorted point in K pixel coordinates
 * @param yn [Output] K-normalized undistorted point
 *
 * @return true on success, false if u1 is not found in the lookup!
 *
 *
 */
    bool operator()(const cvl::Vector<T,2>& ykd,
                    cvl::Vector<T,2>& yn){

        Vector2<T> Ly=L*ykd;
        int lookup_row = (int)round(Ly(0));
        int lookup_col = (int)round(Ly(1)); // curiously I think this will cause a bias of 0.5 pixels...
        if(lookup.is_in(lookup_row,lookup_col)){
            Taylor1<T> t=lookup(lookup_row, lookup_col);
            if(!t.valid) return false;
            yn = t.f + t.J * (Kinv*ykd - t.a);
        }


        return true;
    }


private:

    // Distorted image parameters
    cvl::Matrix<T,3,3> K, Kinv,L,Linv;

    uint rows, cols;
    uint lookup_rows, lookup_cols;



    std::vector<Taylor1<T>> lookup_data;

    MatrixAdapter<Taylor1<T>> lookup;

    bool inited=false;

};










}
