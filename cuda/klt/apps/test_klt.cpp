#include <vector>
#include <set>
#include <map>
#include <thread>
#include <cassert>
#include <cstdio>

#include <QApplication>

#include <mlib/parameter_editor/editor.h>

#include <mlib/opencv_util/imshow.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/opencv_util/draw_arrow.h>
#include <mlib/datasets/stereo_datasets.h>
#include <mlib/utils/mlibtime.h>

#include <mlib/cuda/klt/tracker.h>
#include <iostream>




using std::cout;using std::endl;
using namespace cvl;



//auto dm=buffered_kitti_sequence();
struct TrackerTest:public cvl::Parametrized{
    klt::Tracker daimler_klt=klt::Tracker("daimler klt");
    klt::Tracker kitti_klt=klt::Tracker("kitti klt");
    TrackerTest():cvl::Parametrized("Tracker Test",""){
        add("daimler", daimler_klt.params());
        add("kitti", kitti_klt.params());
    }
};

void track(std::shared_ptr<cvl::BufferedStream<StereoSequenceStream>> seq,
           klt::Tracker& tracker,
           std::string name)
{

    auto s=seq->next();
    mlib::sleep(1);
    mlib::Timer timer(name+"klt1f");
    int n=0;
    while((s=seq->next()))
    {
        cv::Mat1f g=s->grey1f(0);
        timer.tic();
        tracker.track(g);
        timer.toc();        
        imshow(draw_feature_pool(tracker.getFeaturePool(),s->rgb(0)),               name+" klt raw tracks");
        //cout<<timer<<endl;
        waitKey(0);
        if(n++>98) break; // 98
    }
    imshow(draw_feature_pool(tracker.getFeaturePool(),s->rgb(0)),
           name+" klt raw tracks");
    cout<<timer<<endl;
    waitKey(0);
}




int main(int argc, char** argv){
    QApplication                  myApp(argc,argv);

    TrackerTest tt;

    std::string parampath("/home/mikael/co/ip/test_klt.dat");
    std::cout<<"loaded parameters from: "<<parampath<<std::endl;



    cvl::ParameterEditor w;
    w.set(tt.params());
    w.show();

    std::thread thr1f([&](){track(buffered_daimler_sequence(950),tt.daimler_klt,"daimler");});
    std::thread thr1w([&](){track(buffered_kitti_sequence(0),tt.kitti_klt,"kitti");});

    int r=QApplication::exec();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000000000));
    return r;
}
