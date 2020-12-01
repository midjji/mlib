
#if 1
int main(){return  0;}
#else

#include <mlib/vis/Visualization_helpers.h>
#include <mlib/utils/files.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/vis/vis.h>
#include <mlib/utils/simulator_helpers.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <mlib/vis/flow_viewer.h>


using namespace mlib;
using namespace cvl;
using std::cout;using std::endl;
cv::Mat3b get_test_image(bool show_with_cv=false){

    std::string cat="/home/mikael/co/mlib/apps/mlib/data/cat.png";
    assert(fileexists(cat,true));
    cv::Mat3b catim=cv::imread(cat);
    if(show_with_cv){
        cv::imshow("catim",catim);
        cv::waitKey(1000);
    }
    return catim;

}

void test0(std::string name, cv::Mat3b im){
    cvl::vis::showImage(name,im);
}
void test1(cv::Mat3b img) {
    // this experiment does not work well.
    // I must click a image then click exit or it wont work

    int size = 1;
    std::vector<std::thread> thrs;
    for (int i = 0; i < size; ++i) {

        thrs.push_back(std::thread(test0,"catim"+toStr(i), img.clone()));

    }
    for (std::thread& thr:thrs) thr.join();

    cv::waitKey(100);
}
void test2(cv::Mat3b img){
    // half the images are closed then restarted, the other half are overwritten, memory should not grow from this.
    for(int i=0;i<2;++i){
        test1(img);

        for (int i = 0; i < 5; ++i)
            cvl::vis::closeWindow("catim"+toStr(i));
    }
}


void testImshow(){
    // this test tests the capabilities of the vis module
    // the sought behaviour is as follows:
    // 1) thread cross talk
    // 2) show images
    cv::Mat3b im = get_test_image();
    test0("im",im);




}


void testPointClouds(){
    std::vector<Vector3d> pts,cols;
    for(int i=0;i<1000;++i){
        pts.push_back(getRandomUnitVector<double,3>());
        cols.push_back(getRandomUnitVector<double,3>());
    }
    vis::showPointCloud("point cloud 1",pts,cols);

    //cv::waitKey(1000);



    vis::show(pts);
    cv::waitKey(10000);
    std::vector<Vector3d> centers;centers.reserve(100);
    double pi=3.14159265;
    for(double i=0;i<100;++i)
        centers.push_back(Vector3d(cos(2*pi*i/100.0),sin(2*pi*i/100.0),1)*10);
    for(double i=0;i<100;++i)
        centers.push_back(Vector3d(cos(2*pi*i/100.0),sin(2*pi*i/100.0),-1)*10);
    std::vector<PoseD> ps;ps.reserve(100);
    for(auto c:centers)
        ps.push_back(lookAt(Vector3d(0,0,0),c,Vector3d(0,0,1)));
    //ps.push_back(PoseD(c));
    //for(auto p:ps){        cout<<p.getTinW()<<endl;        cout<<p.getT()<<endl;    }
    vis::show(ps);
}


std::vector<Flow> get_bubble(Vector3d origin, Vector3d goal, Vector3d color, double radius, double velocity,int N=20){

    std::vector<Flow> flows;flows.reserve(N);
    for(int i=0;i<N;++i){
        auto o=origin + getRandomUnitVector<double,3>()*radius;
        auto v=(goal - o).normalized()/velocity;
        flows.push_back(Flow(o,v,color));
    }
    return flows;
}




std::vector<Flow> get_flow(){

    std::vector<Flow> flows;flows.reserve(30000);

    for(int i=0;i<1000;++i)
        flows.push_back(Flow(mlib::getRandomUnitVector<double,3>(),Vector3d(0,0,2),Vector3d(1,0,0)));
    for(int i=0;i<900;++i)
        flows.push_back(Flow(mlib::getRandomUnitVector<double,3>()+ Vector3d(1,1,1),Vector3d(0,2,2),Vector3d(0,1,1)));
    for(int i=0;i<1000;++i)
        flows.push_back(Flow(mlib::getRandomUnitVector<double,3>()+ Vector3d(2,0,0),Vector3d(0,0,2),Vector3d(0,1,0)));
    for(int i=0;i<1000;++i)
        flows.push_back(Flow(mlib::getRandomUnitVector<double,3>()+ Vector3d(0,0,2),Vector3d(0,0,2),Vector3d(0,0,1)));
    return flows;
}
PointCloud get_points(){
    std::vector<Vector3d> xs;xs.reserve(1000);
    std::vector<Vector3d> cs;cs.reserve(1000);
    for(int a=0;a<10;++a)
        for(int b=0;b<10;++b)
            for(int c=0;c<10;++c){
                xs.push_back(Vector3d(a,b,c));
                cs.push_back(Vector3d(0,0,1));
            }
    return PointCloud(xs,cs);
}
std::vector<Trajectory> get_trajectories(int N){
    return {};
}
std::shared_ptr<FlowField> get_flow_and_points(){
    std::shared_ptr<FlowField> ff=std::make_shared<FlowField>(get_flow(), get_points(),get_trajectories(0));
    return ff;
}
int main(int, char **)
{
    testImshow();
    testPointClouds();
    show_flow(get_flow(),"just flows");
    show_flow(get_flow_and_points(),"flows and points");
    while(flow_field_viewers_open())
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

}
#endif
