#include <mlib/ceres_util/pose_parametrization.h>
#include <ceres/local_parameterization.h>
namespace cvl {
ceres::LocalParameterization* pose_parametrization()
{
    return new ceres::ProductParameterization(
                new ceres::QuaternionParameterization(),
                new ceres::IdentityParameterization(3));
}
}

