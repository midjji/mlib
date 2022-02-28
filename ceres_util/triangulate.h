#include <vector>
#include <mlib/utils/cvl/pose.h>
#include <mlib/datasets/stereo_calibration.h>

namespace cvl {
Vector3d triangulate(const StereoCalibration& intrinsics,
                     const std::vector<Vector2d>& obs,
                     const std::vector<PoseD>& pvws,
                     Vector3d x,
                     bool reinitialize);
Vector3d triangulate(const StereoCalibration& intrinsics,
                     const std::vector<Vector3d>& obs,
                     const std::vector<PoseD>& pvws,
                     Vector3d x,
                     bool reinitialize);
}
