#include <iostream>
#include <fstream>
#include <cassert>
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
double KittiError::t_err_relative() const{
    return 100.0*t_err;
}
double KittiError::r_err_relative() const{
    return r_err;
}
KittiError::KittiError(PoseD deltaPose, double len, double speed){

    t_err=deltaPose.getT().length()/len;
    r_err=kittiRotationError(deltaPose)/len;
    this->len=len;
    this->speed=speed;
}
std::vector<KittiError>
compute_benchmark_metrics(const DistLsh& distlsh,
                          std::vector<PoseD> gt_poses,
                          std::vector<PoseD> poses,
                          std::vector<double> lengths )
{
    std::vector<KittiError> kes;
    // computes the benchmark metrics

    // this contains all information neccessary to evaluate a single sequence, but not sets of sequences... hmm
    kes.reserve(lengths.size());
    // parameters
    // step size =10 => for every second
    // valid parameter as option? ie get the res for which the len permits equal accuracy? well they dont have so...
    int step_size=1; // should be 1 for proper eval but 10 is faster for testing...
    // for all start positions do

    for (int first_frame=0; first_frame<(int)gt_poses.size(); first_frame+=step_size) {
        // for all segment lengths do
        for (double len : lengths)
        {

            // compute last frame
            int last_frame =distlsh.getIndexPlusDist(first_frame,len);

            // continue, if sequence not long enough
            if (last_frame==-1)
                continue;
            if(last_frame<0 || last_frame>(int)gt_poses.size())  {
                mlog()<<"Wierd: "<<last_frame<<"\n";
                continue;
            }

            PoseD pb_gt   =gt_poses[last_frame];
            PoseD pb      =poses[last_frame];

            // compute speed
            float num_frames = (float)(last_frame-first_frame+1);
            float speed = len/(0.1*num_frames); // in meters per second

            // compute rotational and translational KittiError
            PoseD pose_delta_gt     = gt_poses[first_frame].inverse()*pb_gt;
            PoseD pose_delta_result = poses[first_frame].inverse()*pb;
            PoseD pose_error        = pose_delta_result.inverse()*pose_delta_gt;

            kes.push_back(KittiError(pose_error,len,speed));
        }
    }

    return kes;
}

Result::Result(Sequence& seq,
               std::vector<PoseD> Pwcs,
               std::string name_,
               std::vector<double> lengths):lengths(lengths),name_(name_)
{   
    if(int(Pwcs.size())!=seq.samples()){
        cout<<"wrong number of poses in estimate for seq: "<<seq.name()<<" "<<seq.samples()<<" "<<Pwcs.size()<<endl;
        Pwcs.resize(seq.samples());
    }
    kes=compute_benchmark_metrics(seq.dist_lsh(),seq.gt_poses(),Pwcs,lengths);
}


std::map<std::string, std::vector<double>> Result::graphs() const{
    std::map<std::string, std::vector<double>> grs;

    grs[name()+" translation error by distance"]=getTL(lengths);
    grs[name()+" rotation    error by distance"]=getRL(lengths);
    grs[name()+" translation error by speed   "]=getTS(lengths);
    grs[name()+" rotation    error by speed   "]=getTS(lengths);
    return grs;
}

std::string Result::name() const{
    return name_;
}
double Result::translation_error_average() const{
    double terr=0;
    for(auto ke:kes)
        terr+=ke.t_err_relative();
    terr/=double(kes.size());
    return terr;
}
double Result::rotation_error_average() const{
    double rerr=0;
    for(auto ke:kes)
        rerr+=ke.r_err_relative();
    rerr/=double(kes.size());
    return rerr;
}
Vector2d Result::mean_errs() const{
    return {rotation_error_average(),translation_error_average()};
}

#if 0
/**
      * Compute improved metrics
      *
      * Number of failures as in Prel differs from gtw by more than angle, and translation
      *
      *
      * */
// interframe translation and angle error


interframe_angle_error.clear();interframe_angle_error.reserve(seq.samples());
interframe_translation_error.clear();interframe_translation_error.reserve(seq.samples());
auto gts=seq.gt_poses();
for(uint i=1;i<gts.size();++i){
    PoseD gta=gts[i-1];
    PoseD gtb=gts[i];
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
#endif

void Result::save_benchmark_metrics(std::string path){
    plot_errors(kes,path,name(),lengths);
}




//std::string Result::toMatlabStruct(){}











































}// end kitti namespace
}// end namespace cvl
