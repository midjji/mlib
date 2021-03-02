#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "mlib/utils/files.h"
#include <mlib/utils/string_helpers.h>

 
#include <mlib/utils/cvl/convertopencv.h>
#include <mlib/utils/matlab_helpers.h>
#include <kitti/odometry/orig_gnu_plot.h>

#include <kitti/odometry/eval.h>
using std::cout;using std::endl;

using namespace mlib;
namespace cvl{
namespace kitti{


double KittiError::kittiRotationError(PoseD err){
    Matrix3d R=err.getR();
    double a = R(0,0);
    double b = R(1,1);
    double c = R(2,2);
    double d = 0.5*(a+b+c-1.0);
    return std::acos(std::max(std::min(d,1.0),-1.0));
}
KittiError::KittiError(PoseD deltaPose, double len, double speed){

    t_err=deltaPose.getT().length()/len;
    r_err=kittiRotationError(deltaPose)/len;
    this->len=len;
    this->speed=speed;
}






std::vector<double> getDistanceTraveled(const std::vector<cvl::PoseD>& ps){
    std::vector<double> distances;distances.reserve(ps.size()+1);
    distances.push_back(0);
    for(uint t=1;t<ps.size();++t){
        //these have to be in the kitti structure!
        double relative=(ps[t].getT()-ps[t-1].getT()).length();
        distances.push_back(distances.at(t-1)+relative);
    }
    return distances;
}

DistLsh::DistLsh( std::vector<cvl::PoseD> ps){
    // potentially heavy constructor...
    dists=getDistanceTraveled(ps);

    // assume minimum is 0
    // assume 0-lots per meter
    // actually it might be faster to use less dense buckets well typically there will be zero to three per bucket as is, should be pretty fast
    distmap.resize((int)(dists.back()+2));

    for(std::vector<std::pair<uint,double>>& v:distmap)
        v.reserve(64);
    for(uint i=0;i<dists.size();++i){
        distmap.at((uint)dists[i]).push_back(std::make_pair(i,dists[i]));
    }
}

int DistLsh::getIndexPlusDist(uint index, double dist) const{

    if(index>dists.size()) return -2;
    double a=dists.at(index);
    double b=a+dist;
    int bindex=int(std::floor(b)-1);
    if(bindex<0) bindex=0;
    // for each possible index position test in order returning the first match or if all fail return -1
    for(uint m=bindex;m<distmap.size();++m){ // matches behaviour of original despite the problem it implies if there are long jumps
        for(const std::pair<uint,double>& distv : distmap.at(m))
            if(distv.second>b)
                return distv.first;
    }
    return -1;
}





bool Result::init(std::string path){



    std::string estpath=path+mlib::toZstring(seq.sequence,2)+".txt";




    if(!fileexists(estpath)) {
        state="Missing!";
        return false;
    }
    poses=readKittiPoses(estpath);
//    if(usealt)
  //      altposes=readKittiPoses(altestpath);


    if((int)poses.size()==seq.images ){
        state="OK!";
        return true;
    }
    state="Wrong number of poses:"+toStr((int)poses.size()) + " " + toStr(seq.images);
    return false;

}
std::string Result::getDisplayString(){
    std::stringstream ss;
    ss<<"Result: sequence: "<<toZstring(seq.sequence,2)<<" "<<state;
    return ss.str();
}

void Result::evaluate(){
    if(seq.isTraining())
        compute_benchmark_metrics();

    /**
      * Compute improved metrics
      *
      * Number of failures as in Prel differs from gtw by more than angle, and translation
      *
      *
      * */
    // interframe translation and angle error
    if(seq.isTraining()){

        interframe_angle_error.clear();interframe_angle_error.reserve(seq.gt_poses.size());
        interframe_translation_error.clear();interframe_translation_error.reserve(seq.gt_poses.size());

        for(uint i=1;i<seq.gt_poses.size();++i){
            PoseD gta=seq.gt_poses[i-1];
            PoseD gtb=seq.gt_poses[i];
            PoseD gtd=gta.inverse()*gtb;

            PoseD pa=poses[i-1];
            PoseD pb=poses[i];
            PoseD pd=pa.inverse()*pb;
            PoseD d=gtd.inverse()*pd;

            interframe_angle_error.push_back(d.angle_degrees());
            //  ts.push_back(100*d.getTinW().length());// the actual inter frame error but multiplied up by angle

            if(gtd.getTinW().length()>0.8)
                interframe_translation_error.push_back(100*std::abs(gtd.getTinW().length() - pd.getTinW().length())/gtd.getTinW().length());
        }

        //  std::cout<<"angles:    "<<mean(interframe_angle_error)<<" "<<median(interframe_angle_error)<<" "<<max(interframe_angle_error)<<std::endl;
        //  std::cout<<"distances: "<<mean(interframe_translation_error)<<" "<<median(interframe_translation_error)<<" "<<max(interframe_translation_error)<<std::endl;
    }









    evaluated=true;

}
void Result::save_evaluation(std::string basepath){

    assert(evaluated);


    //save_benchmark_metrics(path+"classic/");

    /**
      * Save the improved metrics
      *
      * Number of failures as in Prel differs from gtw by more than angle, and translation
      *
      *
      * */


}
void Result::save_benchmark_metrics(std::string path){
    // big function ? well soso
    if(seq.gt_poses.size()>0)
        plot_errors(kes,path,seq.name,lengths);
}

void Result::compute_benchmark_metrics(){
    assert(seq.isTraining());


    // this contains all information neccessary to evaluate a single sequence, but not sets of sequences... hmm
    kes.reserve(lengths.size());
    // parameters
    // step size =10 => for every second
    // valid parameter as option? ie get the res for which the len permits equal accuracy? well they dont have so...

    // for all start positions do

    for (int first_frame=0; first_frame<seq.images; first_frame+=step_size) {
        assert(seq.images==(int)seq.gt_poses.size());

        // for all segment lengths do
        for (double len : lengths){

            // compute last frame
            int last_frame =distlsh.getIndexPlusDist(first_frame,len);

            // continue, if sequence not long enough
            if (last_frame==-1)
                continue;
            if(last_frame<0 || last_frame>(int)seq.gt_poses.size())  {
                cout<<"Wierd"<<endl;
                assert(false && "invalid index");
                continue;
            }

            PoseD pb_gt   =seq.gt_poses.at(last_frame);
            PoseD pb      =poses.at(last_frame);

            // compute speed
            float num_frames = (float)(last_frame-first_frame+1);
            float speed = len/(0.1*num_frames); // in meters per second

            // compute rotational and translational KittiError
            PoseD pose_delta_gt     = seq.gt_poses.at(first_frame).inverse()*pb_gt;
            PoseD pose_delta_result = poses.at(first_frame).inverse()*pb;
            PoseD pose_error        = pose_delta_result.inverse()*pose_delta_gt;

            kes.push_back(KittiError(pose_error,len,speed));
        }
    }
    cout<<"done"<<endl;
}


//std::string Result::toMatlabStruct(){}












































}// end kitti namespace
}// end namespace cvl
