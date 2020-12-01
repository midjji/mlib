#include <mlib/utils/files.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/simulator_helpers.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <tum/tum.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/cvl/convertopencv.h>
#include <mlib/cuda/edgels.h>

using namespace mlib;
using namespace cvl;
using std::cout;using std::endl;






int main()
{


    TumDataset tum("/home/mikael/nobackup/server/cvl/mikpe75/tum/driving/");
    tum.init();
    while(true){
        for(uint seq=0;seq<tum.sequences.size();++seq){
            TumSequence sq=tum.getSequence(seq);
            for(uint i=0;i<sq.poses_gt.size();++i){
                cout<<"gt poses: "<<sq.poses_gt.size()<<endl;

                TumTimePoint tp=sq.getTimePoint(i);
                cv::imshow("Tum Left",tp.rgb0);
                cv::imshow("Tum Right",tp.rgb1);
                cv::imshow("Tum Left Disp",tp.disp01/128.0f);
                cv::imshow("Tum Right Disp",tp.disp10/128.0f);

                cv::waitKey(0);

            }
        }

    }
    return 0;
}
