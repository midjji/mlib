#include <mlib/vis/mlib_simple_point_cloud_viewer.h>
#include <iostream>
#include <mlib/utils/mlibtime.h>
#include <mlib/vis/pc_order.h>
using namespace mlib;
using namespace cvl;

int main(){




auto a=mlib::pc_viewer("test");
mlib::sleep(10);
std::cout<<"wait for viewers"<<std::endl;
//mlib::sleep(1000);
mlib::wait_for_viewers();

}
