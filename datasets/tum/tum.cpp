

#include <fstream>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <tum/tum.h>


#include <mlib/utils/files.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/cvl/convertopencv.h>
//#include </home/mikael/tst/CImg/CImg.h>


using std::cout;using std::endl;
namespace cvl{


/**
 * @brief read_pfm
 * @param path
 * @return
 * I am highly suspicious of this thing, but the authors swear by it for their dataset.
 * Dont use for anything else
 */
cv::Mat1f read_pfm(std::string path){

    if(!mlib::fileexists(path,true))        exit(1);



    cv::Mat1f m(540,960);
    cv::Mat1f tmp(540,960);






    FILE *f = fopen(path.c_str(), "r");
    if (!f) {
        // error
    }

    char *line = nullptr;
    size_t n = 0;
    // int i = 0;
    //int c,w,h,s;
    int c;
    int w;
    int h;

    // float* disp;

    if (getline(&line, &n, f) == -1) exit(1);
    c = (line[1] == 'F') ? 3 : 1;

    if (getline(&line, &n, f) == -1) exit(1);
    sscanf(line, "%d %d", &w, &h);

    if (getline(&line, &n, f) == -1) exit(1);

    [[maybe_unused]] int s = atoi(line);




    /*
    while ((getline(&line, &n, f)) != -1) {
        if (i == 0) {
            printf("Hdr:%s\n", line);
            c = (line[1] == 'F') ? 3 : 1;
        } else if (i == 1) {
            sscanf(line, "%d %d", &w, &h);
        } else {
            s = atoi(line);
            break;
        }
        i++;
    }
    */
    //printf("shape=(%d, %d, %d), scale=%d\n", c, w, h,s);

    int read=fread(tmp.data, sizeof(float), w*h*c, f);
    if(read!=w*c*h)
        throw new std::runtime_error("bad pfm size");
    assert(read==w*c*h);// new, may be wrong
    fclose(f);
    cv::flip(tmp , m, 0);
    return m;

}
std::vector<std::pair<PoseD,PoseD>>
read_tum_poses(std::string path){
    std::vector<std::pair<PoseD,PoseD>> ps;

    // read frame // read number

    // read char
    // read 16 numbers
    assert(mlib::fileexists(path,true));
    mlib::fileexists(path,true);
    std::ifstream file(path);
    std::string line;
    while(file>>line){                                  //  cout<<line<<endl;

        int toss;
        file>>toss;
        //cout<<toss<<endl;

        char L;
        Matrix4d M;
        file>>L;
        for(int i=0;i<16;++i)        file>> M(i);
        PoseD Plw(M);
        file>>L;
        for(int i=0;i<16;++i)        file>> M(i);
        PoseD Prw(M);
        ps.push_back(std::make_pair(Plw.inverse(),Prw.inverse()));
    }
    return ps;
}


void TumTimePoint::getImages(std::vector<cv::Mat1w>& images){
    images.clear();
    assert(gray0.rows==gray1.rows);
    assert(gray0.cols==gray1.cols);
    uint rows=gray0.rows;
    uint cols=gray0.cols;
    cv::Mat1w I0(rows,cols);
    cv::Mat1w I1(rows,cols);
    cv::blur(gray0,gray0,cv::Size(3,3));
    cv::blur(gray1,gray1,cv::Size(3,3));
    convertTU(gray0,I0,4096);
    convertTU(gray1,I1,4096);
    images.push_back(I0);
    images.push_back(I1);
}
TumSequence::TumSequence(std::string drivingpath,
            std::string speed,
            std::string direction,
            std::string focallength,Matrix3d K):drivingpath(drivingpath),
    speed(speed),
    direction(direction),
    focallength(focallength),K(K){
    std::string part=focallength + direction + speed;
    std::string path=drivingpath + "camera_data/" + part +"camera_data.txt";
    poses_gt=read_tum_poses(path);
    size=poses_gt.size();
    /*
    PoseD P0w=poses_gt[0].first;
    PoseD P1w=poses_gt[0].second;
    baseline=std::abs((P0w.inverse()*P1w).getT()[0]);
    cout<<endl;
    cout<<P0w<<endl;
    cout<<P1w<<endl;
    std::cout<<baseline<<std::endl;
    */
}

Matrix3d  TumSequence::getK(){return K;}
TumTimePoint TumSequence::getTimePoint(uint index){
    //cout<<"tumtimepoint: "<<index<<endl;
    // left Pose,
    std::string part=focallength+direction+speed;
    std::string path=drivingpath+"frames_cleanpass_webp/"+part;

    TumTimePoint tp;
    tp.index=index;
    tp.K=K;

    tp.P0w=poses_gt[index].first;
    tp.P1w=poses_gt[index].second;

    // 0 is left, 1 is right
    std::string leftpath=path+"left/"+mlib::toZstring(index+1,4)+".webp";
    std::string rightpath=path+"right/"+mlib::toZstring(index+1,4)+".webp";
    mlib::fileexists(leftpath,true);
    mlib::fileexists(rightpath,true);

    tp.rgb0=cv::imread(leftpath);
    tp.gray0=rgb2gray<float>(tp.rgb0);
    tp.rgb1=cv::imread(rightpath);
    tp.gray1=rgb2gray<float>(tp.rgb1);
    assert(tp.rgb0.data!=nullptr);
    assert(tp.rgb1.data!=nullptr);
    cv::Mat1f gray0,gray1;

    leftpath=drivingpath+"disparity/"+part+"left/"+mlib::toZstring(index+1,4)+".pfm";
    rightpath=drivingpath+"disparity/"+part+"right/"+mlib::toZstring(index+1,4)+".pfm";
    mlib::fileexists(leftpath,true);
    mlib::fileexists(rightpath,true);
    tp.disp01=read_pfm(leftpath);
    tp.disp10=read_pfm(rightpath);
    // object path
    std::string object_path_left=drivingpath+"object_index/"+part+"left/"+mlib::toZstring(index+1,4)+".pfm";
    std::string object_path_right=drivingpath+"object_index/"+part+"right/"+mlib::toZstring(index+1,4)+".pfm";

    tp.object_left=read_pfm(object_path_left);
    tp.object_right=read_pfm(object_path_right);
    //for(int r=0;r<tp.object_left.rows;r+=10)            for(int c=0;c<tp.object_left.cols;c+=10)                cout<<tp.object_left(r,c)<<endl;
    //tp.object_left=mlib::normalize01(tp.object_left);
    //tp.object_right=mlib::normalize01(tp.object_right);



    return tp;
}
TumDataset::TumDataset(){}

TumDataset::TumDataset(std::string basepath):basepath(basepath){}

void TumDataset::init(){
    if(!mlib::fileexists(basepath+"driving.txt",true)) exit(1);



    sequences.push_back(TumSequence(basepath,"slow/","scene_forwards/","35mm_focallength/",K35));
    sequences.push_back(TumSequence(basepath,"fast/","scene_forwards/","35mm_focallength/",K35));

    sequences.push_back(TumSequence(basepath,"slow/","scene_backwards/","35mm_focallength/",K35));
    sequences.push_back(TumSequence(basepath,"fast/","scene_backwards/","35mm_focallength/",K35));

    sequences.push_back(TumSequence(basepath,"slow/","scene_forwards/","15mm_focallength/",K15));
    sequences.push_back(TumSequence(basepath,"fast/","scene_forwards/","15mm_focallength/",K15));

    sequences.push_back(TumSequence(basepath,"slow/","scene_backwards/","15mm_focallength/",K15));
    sequences.push_back(TumSequence(basepath,"fast/","scene_backwards/","15mm_focallength/",K15));

}

TumSequence TumDataset::getSequence(uint sequence){
    assert(sequence<sequences.size());
    return sequences.at(sequence);
}
}
