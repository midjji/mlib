#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "mlib/utils/files.h"
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/cvl/convertopencv.h>
#include <mlib/utils/mlibtime.h>
#include <kitti/odometry/kitti.h>

using std::cout;using std::endl;
using namespace mlib;
namespace cvl{

namespace kitti{
/**
cv::Point min_loc, max_loc
bad idea though, it assumes alot of wierd shit
cv::minMaxLoc(your_mat, &min, &max, &min_loc, &max_loc);
**/
template<class T> bool minmax(cv::Mat_<T> im,T& minv, T& maxv){
    assert(im.rows>1);
    assert(im.cols>1);
    minv=maxv=im(0,0);
    if(im.rows==0||im.cols==0)
        return true;

    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c){
            T v=im(r,c);
            if(v>maxv) maxv=v;
            if(v<minv) minv=v;
        }
    return false;
}
template<class T>
cv::Mat1f normalize01(const cv::Mat_<T>& im){
    cv::Mat1f ret(im.rows,im.cols);
    T min,max;min=0;max=1;
    minmax(im, min, max);
    //std::cout<<"cv minmax: "<<min<<" "<<max<<std::endl;
    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c)
            ret(r,c)=((float)(im(r,c)-min))/((float)(max-min));
    return ret;
}

int count_png_in_dir(std::string path){
    int number=0;
    while(true){
        std::string lpath=path+"image_0/"+mlib::toZstring(number,6)+".png";
        std::string rpath=path+"image_1/"+mlib::toZstring(number,6)+".png";
        if(mlib::fileexists(lpath,false) && mlib::fileexists(rpath,false))
            number++;
        else
            return number;
    }
}

bool KittiDataset::checkFiles(){
    init();
    std::vector<cv::Mat1b> imgs;


    // check assumption that k0 = k1 for all sequences! jupp!
    //for(Sequence seq:seqs)        cout<<"k0==k1 for all: "<<(seq.ks[0] -seq.ks[1]).abs().sum() - std::abs(seq.ks[1](0,3))<<endl;





    // check the number of images
    assert([&](){
        for(int seq:sequences)
            assert(seqimgs[seq]==count_png_in_dir(basepath+"sequences/"+toZstring(seq,2)+"/"));
        return true;
    }());


    for(uint seq=0;seq<seqimgs.size();++seq)
        for(int num=0;num<seqimgs[seq];num+=100){

            if(!getImage(imgs,num,seq)){
                //     cout<<"seq: "<<seq<<" "<<num<<endl;
                return false;
            }
        }

    return true;

}
KittiDataset::KittiDataset(std::string basepath):basepath(basepath){init();}
void KittiDataset::init(){
    if(inited) return;
    inited=true;
    for(int seq:sequences){
        Sequence s(basepath,seq,rowss[seq],colss[seq],seqimgs[seq]);
        seqs.push_back(s);
    }
}
std::vector<Sequence> KittiDataset::get_training_sequences(){
    std::vector<Sequence> ss; ss.reserve(11);
    for(int i=0;i<training_sequences;++i)
        ss.push_back(seqs[i]);
    return ss;
}
std::vector<Sequence> KittiDataset::get_joke_sequences(){
    std::vector<Sequence> ss=get_training_sequences();
    for(auto& s:ss)
        s=s.shrunk();
    return ss;
}

std::string KittiDataset::getseqpath(int sequence){
    return basepath+"sequences/"+mlib::toZstring(sequence,2)+"/";
}
bool KittiDataset::getImage(std::vector<cv::Mat1b>& images, int number, int sequence, bool cycle){
    assert(inited);
    // cycle
    if(cycle){
        sequence=sequence % (int)seqimgs.size();
        number=number % (int)seqimgs.at(sequence);
    }

    return seqs[sequence].getImages(images,number);
}

int KittiDataset::images(int sequence){return seqimgs.at(sequence);}


std::shared_ptr<KittiOdometrySample> KittiDataset::get_sample(int sequence, int frameid) {
    Sequence seq= getSequence(sequence);
    return seq.get_sample(frameid);
}
Sequence KittiDataset::getSequence(int index){init();
                                              if(index<0){
                                                  Sequence seq=seqs.at(-index).shrunk();
                                                  return seq;
                                              }
                                              return seqs.at(index);    }
std::shared_ptr<KittiOdometrySample> KittiDataset::next(){
    return get_sample(sequence,index++);
}



void testKitti(std::string basepath, bool withstereo){
    KittiDataset kd(basepath);
    kd.init();

    while(true){
        for(uint seq=0;seq<kd.sequences.size();++seq){
            Sequence sq=kd.getSequence(seq);
            for(int i=0;i<sq.samples();i+=1){
                if(!withstereo){
                    std::vector<cv::Mat1b> imgs;
                    if(!sq.getImages(imgs,i)) continue;
                    cv::imshow("Kitti Left",imgs[0]);
                    cv::imshow("Kitti Right",imgs[1]);
                    cv::waitKey(10);

                }else{

                    std::vector<cv::Mat1w> imgs;
                    cv::Mat1f disp;
                    if(!sq.getImages(imgs,disp,i)) continue;
                    for(int r=0;r<disp.rows;++r)
                        for(int c=0;c<disp.cols;++c){
                            float v=disp(r,c);if(v<-1) v=-1;
                            v+=64;
                            disp(r,c)=v;

                        }
                    disp(0,0)=256-64;


                    cv::imshow("Kitti Left",normalize01(imgs[0]));
                    cv::imshow("Kitti Right",normalize01(imgs[1]));
                    cv::imshow("Disparity", normalize01(disp));
                    cv::waitKey(10);

                }

            }
        }
    }
}




































std::vector<std::vector<PoseD>> trajectories(std::string basepath){
    KittiDataset kd(basepath);
    kd.init();

    std::vector<std::vector<PoseD>> trs; trs.reserve(100);
    for(const auto& seq : kd.seqs){
        trs.push_back(seq.gt_poses());
    }
    return trs;
}








} // end namespace kitt
}// end namespace cvl
