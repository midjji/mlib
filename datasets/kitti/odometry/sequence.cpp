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












bool Sequence::getImages(std::vector<cv::Mat1b>& imgs, int number) const{
    imgs.clear();imgs.reserve(4);
    if(number<0 || number>images) return false;

    std::string lpath=seqpath+"image_0/"+mlib::toZstring(number,6)+".png";
    std::string rpath=seqpath+"image_1/"+mlib::toZstring(number,6)+".png";
    if(!mlib::fileexists(lpath,false)) {cout<<"left image not found: "<<lpath <<endl; assert(false);return false;}
    if(!mlib::fileexists(rpath,false)) {cout<<"right image not found: "<<rpath<<endl; assert(false);return false;}
    cv::Mat1b left = cv::imread(lpath,cv::IMREAD_GRAYSCALE);
    cv::Mat1b right= cv::imread(rpath,cv::IMREAD_GRAYSCALE);

    imgs.push_back(left);
    imgs.push_back(right);

    assert(left.rows==rows);
    assert(left.cols==cols);
    assert(right.rows==rows);
    assert(right.cols==cols);


    return true;
}
PoseD Sequence::getPose(int number) const{    return gt_poses.at(number);}
PoseD Sequence::getPoseRightLeft() const{        return PoseD(-Vector3d(baseline,0,0));    }
bool Sequence::getImages(std::vector<cv::Mat1w>& images,cv::Mat1f& disparity, int number) const{
    std::vector<cv::Mat1b> imgs;
    if(!getImages(imgs,number)) return false;
    // convert to 1w
    cv::Mat1w L(rows,cols);
    cv::Mat1w R(rows,cols);
    for(int row=0;row<rows;++row)
        for(int col=0;col<cols;++col)
            L(row,col)=imgs[0](row,col)*16; // bitshift is faster but it should be converted automatically by the compiler
    images.push_back(L);
    for(int row=0;row<rows;++row)
        for(int col=0;col<cols;++col)
            R(row,col)=imgs[1](row,col)*16; // bitshift is faster but it should be converted automatically by the compiler
    images.push_back(R);

    std::string stereopath=path+"stereo/"+name+"/"+toZstring(number)+".exr";

    if(!mlib::fileexists(stereopath,false)) {cout<<"stereo image not found: "<<stereopath<<endl; assert(false);return false;}
    disparity=cv::imread(stereopath,cv::IMREAD_ANYDEPTH);


    assert(disparity.rows==rows);
    assert(disparity.cols==cols);

    return true;

}

bool Sequence::isTraining(){return sequence<11;}
std::vector<unsigned int> Sequence::getDistantFrames(){
    assert(isTraining());
    // shouldnt be any loop closures in it...
    // so check each pose distance to every other
    std::vector<PoseD>        solitaries; solitaries.reserve(gt_poses.size());
    std::vector<unsigned int> indexes;    indexes.reserve(gt_poses.size());

    for(PoseD p:gt_poses){
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
cv::Mat1b Sequence::getPoseConfusionMatrix(){
    // the distance in translation and rotation from every image to every other...
    cv::Mat1b im(gt_poses.size(),gt_poses.size());
    for(uint i=0;i<gt_poses.size();++i){
        for(uint j=0;j<gt_poses.size();++j){
            double dist=255-std::min(255.0,2.55*(gt_poses[i].distance(gt_poses[j])));
            // double rdist=255-std::min(255.0,255*std::abs((gt_poses[i].inverse()*(gt_poses[j])).getAngle())/1.5);
            im(i,j)=dist;
            //im(i,j)=cv::Vec3b(dist,0,rdist);
        }
    }
    return im;
}
cv::Mat3b Sequence::getMap(){
    std::vector<Vector3d> tws;
    tws.reserve(gt_poses.size());
    std::vector<double> xs,ys,zs;
    xs.reserve(gt_poses.size());
    ys.reserve(gt_poses.size());
    zs.reserve(gt_poses.size());
    for(uint i=0;i<gt_poses.size();++i){
        Vector3d tw=gt_poses[i].translation(); // kitti is in inversem, x,z is interesting
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


cvl::Matrix3d Sequence::getK(){
    if((ks[0].getBlock<0,0,3,3>()-ks[1].getBlock<0,0,3,3>()).abs().sum()>1e-7) cout<<"Different K for L,R "<<endl;
    return ks[0].getBlock<0,0,3,3>();
}






Sequence::Sequence(std::string path_, int sequence_,int rows_, int cols_, int images_){
    path=path_;
    sequence=sequence_;
    name=toZstring(sequence,2);
    rows=rows_;
    cols=cols_;
    images=images_;
    seqpath=path+"sequences/"+name+"/";
}
void Sequence::readSequence(){
    std::string seqpath=path+"sequences/"+name+"/";


    times=readTimes(seqpath+"times.txt");
    if(sequence<11){
        gt_poses=readKittiPoses(path+"poses/"+name+".txt");
        assert((int)gt_poses.size()==images);
    }

    ks=readCalibrationFile(seqpath+"calib.txt");
    assert(ks.size()==5);

    baseline=-ks[1](0,3)/ks[1](0,0);
    assert(baseline>0);


}














std::vector<cvl::PoseD> readKittiPoses(std::string path){
    //  cout<<"readKittiPoses: "<<path<<endl;
    // verify path
    if(!fileexists(path)){
        throw new std::logic_error(path + "file not found ");
    }
    // read file
    std::vector<double> vals;double val; vals.reserve(10000000);
    {
        std::ifstream fin; fin.open(path);
        while(fin>>val) vals.push_back(val);
        fin.close();
    }

    // parse file
    if(vals.size()==0)          throw new std::logic_error("Failure to parse data - no doubles"+path);


    if(vals.size()%12!=0)       throw new std::logic_error("Failure to parse data - to few values"+path);
    std::vector<cvl::PoseD> poses;poses.reserve(vals.size()/12+1);
    for(uint i=0;i<vals.size();i+=12){
        cvl::PoseD pose(cvl::Matrix34d(vals[i],vals[i+1],vals[i+2],vals[i+3],
                vals[i+4],vals[i+5],vals[i+6],vals[i+7],
                vals[i+8],vals[i+9],vals[i+10],vals[i+11]));
        pose.normalize();
        poses.push_back(pose);
    }
    //assert(validPoses(poses));

    return poses;
}

void writeKittiPoses(std::string name, std::vector<PoseD> ps){


    //  saferSystemCall("mkdir -p "+dir);
    std::stringstream ss;
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
    std::ofstream ofs;
    ofs.open(name);
    ofs<<ss.str()<<endl;
    ofs.close();
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

std::vector<Matrix34d> readCalibrationFile(std::string path  ){
   // cout<<"readCalibrationFile path: "<<path<<endl;
    std::vector<Matrix34d> ks;ks.reserve(4);
    assert(fileexists(path,true));

    std::ifstream ifs;ifs.open(path);


    std::vector<double> vals;vals.reserve(9);
    bool test=true;
    for(int i=0;i<5;++i){
        vals.clear();vals.resize(12);
        char skip;

        for(int j=0;j<3;++j){
            test=test && ifs>>skip;
        }
        for(int j=0;j<12;++j){
            test=test && ifs>>vals[j];
        }
        assert(test);
        Matrix34d K=Matrix34d::copy_from(&vals[0]);

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
