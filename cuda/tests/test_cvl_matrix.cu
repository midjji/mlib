/**
 * This file should include all the .cuh headers which contain any cuda test at all.
 * Tests are written in headers only, not sure why it works atm but it does and is preferable
 * Tests are written with a suffix to remove code clutter and unnec includes in the definition files
 * This file is part of the cmake/common/use_gtest()
 * Example:
 * have this file, include tests, and make will automatically run all tests verbosely
 *
 */

#include <mlib/utils/cvl/pose.h>
using namespace cvl;
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>



template<class T>
__global__ void check_matrix(Matrix3<T> a){
    Matrix3<T> b=a;
    T d=b.sum();
}


TEST_CASE("KERNEL_COMPILES_CHECK,MATRIX"){
    dim3 grid(100,100,1);
    dim3 threads(32,1,1);
    Matrix3d a(1,2,3,4,5,6,7,8,9);
    check_matrix<<<grid,threads>>>(a);
    //Matrix3x3<std::complex<double>> c(1,2,3,4,5,6,7,8,9);
    //check_matrix<<<grid,threads>>>(c);

}
__global__ void check_pose(PoseD a){
    PoseD b=a;
    double d=std::abs(b.getT().length());
}
TEST_CASE("KERNEL_COMPILES_CHECK,POSE"){
    dim3 grid(100,100,1);
    dim3 threads(32,1,1);
    PoseD a;
    check_pose<<<grid,threads>>>(a);

}





