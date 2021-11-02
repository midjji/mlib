#include <iostream>

#include <mlib/datasets/kitti/odometry/kitti.h>
#include <mlib/cuda/mbm.h>
#include <opencv2/highgui.hpp>
#include <mlib/utils/cvl/convertopencv.h>


using std::cout;using std::endl;
using namespace cvl;



void printhelp(const char** argv){
    cout<<"Usage: "<<argv[0]<<" <dataset type> <path to dataset> ";
    cout<<"example: if the data is organized as follows: \n /ssdcache/datasets/kitti/dataset/sequences/00/image_0/\n";
    cout<<"argument 2 should be /ssdcache/datasets/kitti/dataset/"<<endl;
    cout<<"example:\n "<<argv[0]<<" kitti /ssdcache/datasets/kitti/dataset/ "<<endl;
}


void testKittiStereo(kitti::KittiDataset kd, bool testing);

void testbmstereo(kitti::KittiDataset kd);

int test()
{
    std::string path="/home/mikael/datasets/kitti/odometry/";
    testKittiStereo(cvl::kitti::KittiDataset(path),true);
    return 0;
}


int main(int argc, const char** argv){


    return test();
    //return example();





    return 0;
}



void testKittiStereo(kitti::KittiDataset kd, bool testing){

    // main loop
    {

        cout<<"mbm"<<endl;


        // max size is fixed!

        std::vector<cv::Mat1b> imgs;


        for(auto seq:kd.seqs)
        {

            cout<<"seq:"<<seq->name()<<endl;

            cvl::MBMStereoStream mbm;
            // cvl::TemporalStereoStream tss;
            mbm.init(64,seq->rows(),seq->cols());

            // tss.init(seq.rows,seq.cols,cvl::Matrix<float,3,3>(seq.getK()),seq.baseline);

            cv::Mat1f previousdispf;
            bool first=true;
            for(int i=0;i<seq->samples();i+=1)
            {

                if(!seq->getImages(imgs,i)) return;
                //cout<<"read images"<<endl;
                cv::Mat1b left=imgs[0];
                cv::Mat1b right=imgs[1];



                cv::Mat1b disp=mbm(left,right);
                mbm.displayTimers();
                cv::imshow("disp0",2*disp);
                cv::imshow("disp1",left);

                cv::Mat1f dispf=cvl::toMat_<float,unsigned char>(disp);

                if(first){
                    first=false;
                }else{
                    //    cv::Mat1f im=tss(previousdispf,dispf,PoseD());
                    //    im=mlib::normalize01(im);
                    //   cv::imshow("refined disp",im);
                    cout<<"image: "<<i<<endl;
                    cv::waitKey(0);
                }

                previousdispf=dispf;


                char key=cv::waitKey(30);
                if(key=='q') break;
                if(testing && (i>300)) break;
            }
            if(testing) break;
        }
    }
    cout<<"mbm - done"<<endl;
}
