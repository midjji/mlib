#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <mlib/utils/files.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/convertopencv.h>
#include <kitti/odometry/sequence.h>

using namespace mlib;
using std::cout;using std::endl;
namespace cvl{
namespace kitti{

int Sequence::rows() const{return rows_;}
int Sequence::cols() const{return cols_;}
std::string Sequence::seqpath() const{
    std::string tmp=path_+"sequences/"+name()+"/";
    return tmp;
}
int Sequence::sequence() const{return sequence_;}
std::string Sequence::name() const{ return toZstring(sequence_,2);}
std::string Sequence::description() const{return description_;}
double Sequence::baseline() const{return baseline_;}
double Sequence::fps() const{return 10.0;}
Fid2Time Sequence::fid2time() const
{
    std::vector<double> ts;ts.reserve(times_.size());

    for(int i=0;i<int(times_.size());++i){
        ts.push_back(i*0.1);
    }
    auto f2t=Fid2Time(ts);
    return f2t;
}
std::shared_ptr<KittiOdometrySample> Sequence::get_sample(int index) const{
    std::vector<cv::Mat1w> images;
    cv::Mat1f disparity;

    if(getImages(images,disparity,index)){

        return std::make_shared<KittiOdometrySample>(images,disparity,sequence(),index, times().at(index));
    }
    mlog()<<"tried to read sample out of bounds"<<index<<" " <<samples_<<"\n";
    return nullptr;
}




bool Sequence::getImages(std::vector<cv::Mat1b>& imgs, int number) const{
    imgs.clear();imgs.reserve(4);
    if(number<0 || number>samples_) return false;

    std::string lpath=seqpath()+"image_0/"+mlib::toZstring(number,6)+".png";
    std::string rpath=seqpath()+"image_1/"+mlib::toZstring(number,6)+".png";
    if(!mlib::fileexists(lpath,false)) {cout<<"left image not found: "<<lpath <<endl; assert(false);return false;}
    if(!mlib::fileexists(rpath,false)) {cout<<"right image not found: "<<rpath<<endl; assert(false);return false;}
    cv::Mat1b left = cv::imread(lpath,cv::IMREAD_GRAYSCALE);
    cv::Mat1b right= cv::imread(rpath,cv::IMREAD_GRAYSCALE);

    imgs.push_back(left);
    imgs.push_back(right);

    if(left.rows!=rows()) mlog()<<"missmatch: "<<left.rows<<" "<<rows()<<"for id"<<number<<"\n";
    if(right.rows!=rows()) mlog()<<"missmatch: "<<right.rows<<" "<<rows()<<"for id"<<number<<"\n";

    if(left.cols!=cols()) mlog()<<"missmatch: "<<left.cols<<" "<<cols()<<"for id"<<number<<"\n";
    if(right.cols!=cols()) mlog()<<"missmatch: "<<right.cols<<" "<<cols()<<"for id"<<number<<"\n";
    return true;
}
PoseD Sequence::getPose(int number) const{    return gt_poses_[number];}
PoseD Sequence::gt_pose(int number) const{    return gt_poses_[number];}
PoseD Sequence::getPoseRightLeft() const{        return PoseD(-Vector3d(baseline(),0,0));    }
bool Sequence::getImages(std::vector<cv::Mat1w>& images,cv::Mat1f& disparity, int number) const{
    std::vector<cv::Mat1b> imgs;
    if(!getImages(imgs,number)) return false;
    // convert to 1w
    cv::Mat1w L(rows(),cols());
    cv::Mat1w R(rows(),cols());
    for(int row=0;row<rows();++row)
        for(int col=0;col<cols();++col)
            L(row,col)=imgs[0](row,col)*16; // bitshift is faster but it should be converted automatically by the compiler
    images.push_back(L);
    for(int row=0;row<rows();++row)
        for(int col=0;col<cols();++col)
            R(row,col)=imgs[1](row,col)*16; // bitshift is faster but it should be converted automatically by the compiler
    images.push_back(R);

    std::string stereopath=path_+"stereo/"+name()+"/"+toZstring(number)+".exr";

    if(!mlib::fileexists(stereopath,false)) {cout<<"stereo image not found: "<<stereopath<<endl; assert(false);return false;}
    disparity=cv::imread(stereopath,cv::IMREAD_ANYDEPTH);


    assert(disparity.rows==rows());
    assert(disparity.cols==cols());

    return true;

}
std::vector<double> Sequence::times() const{
    return times_;
}
std::vector<PoseD> Sequence::gt_poses() const{return gt_poses_;} // Pwc(t)
int Sequence::samples() const{return samples_;}
bool Sequence::is_training() const{return sequence()<11;}
std::vector<unsigned int> Sequence::getDistantFrames(){
    assert(is_training());
    // shouldnt be any loop closures in it...
    // so check each pose distance to every other
    std::vector<PoseD>        solitaries; solitaries.reserve(gt_poses_.size());
    std::vector<unsigned int> indexes;    indexes.reserve(gt_poses_.size());

    for(PoseD p:gt_poses_){
        bool solitary =true;
        p=p.inverse();
        for(PoseD sol:solitaries){
            PoseD tmp=p.inverse()*sol;
            if(tmp.translation().length()<1) {solitary=false;break;}
            if(tmp.angle_degrees()<10) {solitary=false;break;}
        }
        if(solitary)
            solitaries.push_back(p);
    }
    return indexes;
}
std::vector<PoseD> Sequence::gt_vehicle_poses() const{
    std::vector<PoseD> ps=gt_poses_;
    // Should be Pwc*Pcv = Pwv

    for(auto& p:ps){        p=p*P_camera_vehicle();    }

    return ps;
}
cv::Mat1b Sequence::getPoseConfusionMatrix(){
    // the distance in translation and rotation from every image to every other...
    int rows=samples();
    int cols=samples();
    cv::Mat1b im(rows,cols,uchar(0));
    if(!is_training()) return im;
    for(int i=0;i<samples();++i){
        for(int j=0;j<samples();++j){
            double dist=255-std::min(255.0,2.55*(gt_poses_[i].distance(gt_poses_[j])));
            // double rdist=255-std::min(255.0,255*std::abs((gt_poses[i].inverse()*(gt_poses[j])).getAngle())/1.5);
            im(i,j)=dist;
            //im(i,j)=cv::Vec3b(dist,0,rdist);
        }
    }
    return im;
}
cv::Mat3b Sequence::getMap(){
    std::vector<Vector3d> tws;
    tws.reserve(samples());
    std::vector<double> xs,ys,zs;
    xs.reserve(samples());
    ys.reserve(samples());
    zs.reserve(samples());
    for(int i=0;i<samples();++i){
        Vector3d tw=gt_poses_[i].translation(); // kitti is in inversem, x,z is interesting
        std::swap(tw[1],tw[2]);
        tws.push_back(tw);
        xs.push_back(tw[0]);
        ys.push_back(tw[1]);
        zs.push_back(tw[2]);
    }
    // get min max of each
    double xminv,yminv,zminv,xmaxv,ymaxv,zmaxv;
    minmax(xs, xminv, xmaxv);
    minmax(ys, yminv, ymaxv);
    minmax(zs, zminv, zmaxv);

    Vector3d minv(xminv,yminv,zminv);
    Vector3d maxv(xmaxv,ymaxv,zmaxv);
    // poses in meters

    auto v=maxv-minv;
    double scale=std::max(v[0],v[1]);


    std::vector<cv::Scalar> cols;
    for(Vector3d& tw:tws){
        tw[0]-=minv[0];
        tw[1]-=minv[1];


        tw[0]/=scale;
        tw[1]/=scale;
        if(tw[2]<0){
            tw[2]/=minv[2];
            tw[2]+=0.5;
            tw[2]*=255;
            cols.push_back(cv::Scalar(tw[2],0,0,0));
        }
        else{
            tw[2]/=maxv[2];
            tw[2]+=0.5;
            tw[2]*=255;
            cols.push_back(cv::Scalar(0,0,tw[2],0));
        }

        //4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02
        //-7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02
        //9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01



        tw[0]*=1000;
        tw[1]*=1000;

        tw[0]+=100;
        tw[1]+=100;
        tw[1]=1200-tw[1];

    }

    // x,y values between 0,1
    cv::Mat3b im=cv::Mat3b::zeros(1200,1200);
    cv::circle(im,cv::Point2f(tws[0][0],tws[0][1]),5,cv::Scalar(0,255,0,0));
    for(uint i=1;i<tws.size();++i){
        cv::line(im,cv::Point2f(tws[i-1][0],tws[i-1][1]),cv::Point2f(tws[i][0],tws[i][1]),cols[i],2);
    }



    return im;
}


cvl::Matrix3d Sequence::getK() const{
    if((ks[0].getBlock<0,0,3,3>()-ks[1].getBlock<0,0,3,3>()).abs().sum()>1e-7)
    {
        mlog()<<"Different K for L,R THAT SHOULD NOT HAPPEN! EXITING DUE TO DATACORRUPTION\n";
        exit(1);
    }
    return ks[0].getBlock<0,0,3,3>();
}
PoseD Sequence::P_camera_vehicle() const{
    return P_camera_vehicle_;
}


namespace  {
bool warned_on_kitti_times=false;
}


Sequence::Sequence(std::string path_,
                   int sequence_,
                   int rows_,
                   int cols_,
                   int samples_):
    path_(path_), sequence_(sequence_),rows_(rows_),cols_(cols_),samples_(samples_){

    times_=readTimes(seqpath()+"times.txt");
    {
        // problem is these are difficult to use
        if(!warned_on_kitti_times){
            warned_on_kitti_times=true;
            cout<<"approximate kitti times"<<endl;
        }
        std::vector<double> ts; ts.reserve(gt_poses().size());
        for(int i=0;i<int(times_.size());++i)
            ts.push_back(0.1*i);
        times_=ts;
    }


    gt_poses_.resize(times_.size());
    if(sequence_<11)
    {
        gt_poses_=readKittiPoses(path_+"poses/"+name()+".txt");
        if(int(gt_poses_.size())!=samples())
            mlog()<<"Configuration missmatch\n";
    }


    ks=readCalibrationFile(seqpath()+"calib.txt");
    assert(ks.size()==5);

    baseline_=-ks[1](1,3)/ks[1](1,0);
    P_camera_vehicle_=PoseD::Identity();

    // P_camera_vehicle_=PoseD(getRotationMatrixY(4 * 3.1415/180.0)*getRotationMatrixX(31.0 * 3.1415/180.0)*getRotationMatrixY(23 * 3.1415/180.0));
    //P_camera_vehicle_ = PoseD(Vector4d(0.99897597013522099, -0.0083444056543999345, 0.0017357290603424664, 0.044433874817507608),Vector3d(0,0,0));
    //P_camera_vehicle_=PoseD(getRotationMatrixZ(0.5*3.1415))*PoseD(ks[4]);
    //P_camera_vehicle_ = PoseD(Vector4d(0.99959445390732238, 0.013343332682150745, 0.025157000775769104, 9.2209667494754373e-05),Vector3d(0,0,0));
    //P_camera_vehicle_ = PoseD(Vector4d(0.644979, -0.146418, 0.471854, -0.583025),Vector3d(0.248715, 0.0942763, -0.046136));
    double f=3.1415/180.0;
    // 1,-1.7 kinda makes sense, its at the back of the car...
    P_camera_vehicle_=PoseD(getRotationMatrixX(1.0*f)*getRotationMatrixY(-0.75*f))*PoseD(Vector3d(0,-1.35,1.01));

    inited=true;
}


Sequence Sequence::shrunk(int newsize) const{
    Sequence seq=*this;
    seq.samples_=std::min(newsize,samples_);
    seq.times_.resize(seq.samples_);
    seq.gt_poses_.resize(seq.samples_);
    return seq;
}

DistLsh Sequence::dist_lsh(){
    if(!distlsh)
        distlsh=DistLsh(gt_poses());
    return distlsh;
}

std::vector<cvl::PoseD> readKittiPoses(std::string path, bool require_found){
    //  cout<<"readKittiPoses: "<<path<<endl;
    // verify path
    if(!fileexists(path)){
        mlog()<<path<< "not found\n";
        if(require_found)
            exit(1);
    }
    // read file
    std::vector<double> vals;double val; vals.reserve(100000);
    {
        std::ifstream fin; fin.open(path);
        while(fin>>val) vals.push_back(val);
    }
    // parse file
    if(vals.empty()) mlog()<<"no data found\n";
    if(vals.size()%12!=0)       mlog()<<"incorrect data format"<<vals.size()<<"\n";
    std::vector<cvl::PoseD> poses;poses.reserve(vals.size()/12);
    for(uint i=0;i<vals.size();i+=12)
    {
        cvl::Matrix34d M(vals[i],vals[i+1],vals[i+2],vals[i+3],
                vals[i+4],vals[i+5],vals[i+6],vals[i+7],
                vals[i+8],vals[i+9],vals[i+10],vals[i+11]);

        cvl::PoseD pose(M);
        poses.push_back(pose);
    }
    return poses;
}

void writeKittiPoses(std::string name, std::vector<PoseD> ps){


    //  saferSystemCall("mkdir -p "+dir);
    std::stringstream ss;
    ss.precision(19);
    for(PoseD p:ps){
        if(!p.is_normal()) cout<<"bad pose when writing kitti"<<p<<endl;
        p.normalize();
        if(!p.is_normal()) cout<<"bad pose when writing kitti 2"<<p<<endl;


        Matrix4d M=p.get4x4();
        if(!M.isnormal()) cout<<"bad pose matrix when writing kitti\n"<<p<<"\n " <<M<<endl;
        for(int row=0;row<3;++row)
            for(int col=0;col<4;++col)
                ss<<M(row,col)<<" ";
        ss<<"\n";
    }
    std::ofstream ofs(name);
    ofs<<ss.str()<<endl;
}

std::vector<double> readTimes(std::string path){
    std::vector<double> times;
    assert(fileexists(path,true));
    std::ifstream ifs;
    double t;
    ifs.open(path);
    while(ifs>>t)
        times.push_back(t);
    ifs.close();




    return times;
}




std::vector<Matrix34d>
readCalibrationFile(std::string path  )
{

    // the k on disk for the right camera matrix looks like:
    //7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02
    //0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00
    //0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
    // the row col measurement one looks like:
    // the row, col one
    //Matrix3d K{0, 2261.54, 1108.15,
    //           2267.22, 0, 519.169,
    //           0, 0, 1}; // for both cameras
    // So the kitti matrixes assume col, row, so convert it.

    Matrix3d R(0,1,0,
               1,0,0,
               0,0,1);




    // cout<<"readCalibrationFile path: "<<path<<endl;
    std::vector<Matrix34d> ks;ks.reserve(4);
    assert(fileexists(path,true));

    std::ifstream ifs;ifs.open(path);
    std::vector<double> vals;vals.reserve(12);
    bool test=true;
    for(int i=0;i<5;++i)
    {
        vals.clear();vals.resize(12);
        char skip;
        for(int j=0;j<3;++j){
            test=test && ifs>>skip;
        }
        for(int j=0;j<12;++j){
            test=test && ifs>>vals[j];
        }
        assert(test);
        Matrix34d K=R*Matrix34d::copy_from(&vals[0]);

        ks.push_back(K);
    }

    ifs.close();
    // check that left and right match!
    //auto M=ks[0]-ks[1];
    //double err=M.abs().sum();
    //if(err>1e-10)
    //    cout<<err<<"\n"<<ks[0]<<"\n"<<ks[1]<<endl;
    //cout<<__PRETTY_FUNCTION__ <<": done"<<endl;
    assert(test);
    return ks;
}




}// end kitti namespace
}// end namespace cvl
