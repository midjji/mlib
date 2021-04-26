#include <mlib/vis/mlib_simple_point_cloud_viewer.h>
#include <mlib/utils/cvl/lookat.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/mlibtime.h>
using namespace mlib;
using namespace cvl;
int main(){
Vector3d up(0,1,0);
Vector3d point(0,0,1);
Vector3d from(0,0,-1);
PoseD P=lookAt(point, from, up);
std::vector<PoseD> ps;ps.push_back(P);
mlog()<<"here\n";
mlib::pc_viewer("test")->setPointCloud(ps);
mlog()<<"here\n";
mlib::wait_for_viewers();
mlog()<<"here\n";
mlib::sleep(10);
return 0;
}
