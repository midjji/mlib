//find -iname '*.h*'|grep -v cuda|grep -v extern >tmp.txt, then just search and replace...
// can I directly generate this file? probably...

#include "mlib/plotter/plot.h"
#include "mlib/opencv_util/cv.h"
#include "mlib/opencv_util/imshow.h"
#include "mlib/opencv_util/draw_arrow.h"
#include "mlib/opencv_util/stereo.h"
#include "mlib/opencv_util/anms.h"
#include "mlib/tests/test_p4p.h"
#include "mlib/tests/datafile.h"
#include "mlib/sfm/p3p/pnp_ransac.h"
#include "mlib/sfm/p3p/parameters.h"
#include "mlib/sfm/p3p/lambdatwist/refine_lambda.h"
#include "mlib/sfm/p3p/lambdatwist/solve_cubic.h"
#include "mlib/sfm/p3p/lambdatwist/lambdatwist.p3p.h"
#include "mlib/sfm/p3p/lambdatwist/solve_eig0.h"
#include "mlib/sfm/p3p/lambdatwist/p3p_timers.h"
#include "mlib/sfm/p3p/p4p.h"
#include "mlib/sfm/camera/distortion_map.h"
#include "mlib/sfm/anms/util.h"
#include "mlib/sfm/anms/draw.h"
#include "mlib/sfm/anms/base.h"
#include "mlib/sfm/anms/grid.h"
#include "mlib/sfm/solvers/geometry_tools.h"
#include "mlib/sfm/solvers/essential_matrix_hartley_gpl/ematrix_hartley_gpl.h"
#include "mlib/sfm/solvers/essential_matrix_solver.h"
#include "mlib/sfm/solvers/motion_tester.h"
#include "mlib/datasets/kitti/mots/dataset.h"
#include "mlib/datasets/kitti/mots/sample.h"
#include "mlib/datasets/kitti/mots/calibration.h"
#include "mlib/datasets/kitti/odometry/eval.h"
#include "mlib/datasets/kitti/odometry/orig_gnu_plot.h"
#include "mlib/datasets/kitti/odometry/sequence.h"
#include "mlib/datasets/kitti/odometry/matlab_plot.h"
#include "mlib/datasets/kitti/odometry/ng_helpers.h"
#include "mlib/datasets/kitti/odometry/kitti.h"
#include "mlib/datasets/kitti/odometry/result.h"
#include "mlib/datasets/daimler/dataset.h"
#include "mlib/datasets/daimler/sample.h"
#include "mlib/datasets/daimler/database.h"
#include "mlib/datasets/daimler/calibration.h"
#include "mlib/datasets/daimler/results_types.h"
#include "mlib/datasets/tum/tum.h"
#include "mlib/doc/mainpage4doxygen.h"
#include "mlib/vis/CameraMatrixManipulator.h"
#include "mlib/vis/flow_viewer.h"
#include "mlib/vis/GLTools.h"
#include "mlib/vis/nanipulator.h"
#include "mlib/vis/convertosg.h"
#include "mlib/vis/fbocamera.h"
#include "mlib/vis/CvGL.h"
#include "mlib/vis/mlib_simple_point_cloud_viewer.h"
#include "mlib/vis/manipulator.h"
#include "mlib/utils/mlibtime.h"
#include "mlib/utils/colormap.h"
#include "mlib/utils/files.h"
#include "mlib/utils/stream.h"
#include "mlib/utils/colormap_tables/jet_colormap.h"
#include "mlib/utils/real_fixpoint.h"
#include "mlib/utils/real.h"
#include "mlib/utils/random.h"
#include "mlib/utils/numerics.h"
#include "mlib/utils/serialization.h"
#include "mlib/utils/memmanager.h"
#include "mlib/utils/common_pragmas.h"

#include "mlib/utils/bounding_box.h"
#include "mlib/utils/mlog/log.h"
#include "mlib/utils/continuous_image.h"
#include "mlib/utils/histogram.h"

#include "mlib/utils/string_helpers.h"
#include "mlib/utils/mzip/generator.h"
#include "mlib/utils/mzip/range.h"
#include "mlib/utils/mzip/mzip_view.h"
#include "mlib/utils/informative_asserts.h"
#include "mlib/utils/workerpool.h"
#include "mlib/utils/checksum.h"
#include "mlib/utils/vector.h"
#include "mlib/utils/simulator_helpers.h"

#include "mlib/utils/interpolation.h"
#include "mlib/utils/buffered_dataset_stream.h"
#include "mlib/utils/cvl/matrix_adapter.h"

#include "mlib/utils/cvl/triangulate.h"

#include "mlib/utils/cvl/epipolar_geometry.h"
#include "mlib/utils/cvl/convertopencv.h"
#include "mlib/utils/cvl/matrix.h"
#include "mlib/utils/cvl/pose.h"
#include "mlib/utils/cvl/tensor.h"
#include "mlib/utils/cvl/converteigen.h"

#include "mlib/utils/cvl/triangulate_nl.h"
#include "mlib/utils/cvl/rotation_helpers.h"
#include "mlib/utils/cvl/quaternion.h"
#include "mlib/utils/cvl/polynomial.h"

#include "mlib/utils/cvl/syncque.h"
#include "mlib/utils/configuration.h"
#include "mlib/utils/sys.h"
#include "mlib/utils/matlab_helpers.h"
#include "mlib/utils/syncmap.h"
#include "mlib/utils/limited_history.h"
#include "mlib/utils/argparser.h"
#include "mlib/utils/binary_search.h"
#include "mlib/utils/smooth_trajectory.h"

#include "mlib/utils/smart_data.h"
#include "mlib/utils/random_vectors.h"
//#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
int main(){return 0;}

