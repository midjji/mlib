#include <opencv2/imgcodecs.hpp>

#include <kitti/odometry/ng_helpers.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/files.h>
#include <mlib/utils/cvl/convertopencv.h>
#include <fstream>

using mlib::toZstring;

namespace cvl{
namespace kitti{
using std::endl;using std::cout;
namespace {
void saferSystemCall(std::string cmd){

    int err=system(cmd.c_str());
    if(err!=0){
        std::ostringstream ss("");
        ss << err;

        throw new std::runtime_error("Failed system call:\n Error: "+ss.str()+"\n "+cmd+"\n");
    }
}
}

std::string calibdata(Matrix34d K,int rows, int cols){

    double baselinemm=1000*K(0,3)/K(0,0);

    std::stringstream ss;
    ss<<"[INTERNAL]\n";
    ss<<"NX   =       "<<cols<<"  # Image dimension in pixel X(U)\n";
    ss<<"NY   =       "<<rows<<"  # Image dimension in pixel Y(V)\n";
    ss<<"FC_U = "<<K(0,0)<<"  # Focal length in pixel (U)\n";
    ss<<"FC_V = "<<K(1,1)<<"  # Focal length in pixel (V)\n";
    ss<<"CC_U = "<<K(0,2)<<"  # Principle point in pixel (U)\n";
    ss<<"CC_V = "<<K(1,2)<<"  # Principle point in pixel (V)\n";
    ss<<"SKEW =   0.000000  # Skew factor\n";
    ss<<"KC_1 =   0.0       # Distortion factor\n";
    ss<<"KC_2 =   0.0       # Distortion factor\n";
    ss<<"KC_3 =   0.0       # Distortion factor\n";
    ss<<"KC_4 =   0.0       # Distortion factor\n";
    ss<<"KC_5 =   0.0       # Distortion factor\n";

    ss<<"[EXTERNAL]\n";
    ss<<"TC_X =   "<<baselinemm<<"  # Translation X [mm]\n";
    ss<<"TC_Y =   0.000000  # Translation Y [mm]\n";
    ss<<"TC_Z =   0.000000  # Translation Z [mm]\n";
    ss<<"RC11 =   1.000000  # Rotation matrix\n";
    ss<<"RC12 =   0.000000\n";
    ss<<"RC13 =   0.000000\n";
    ss<<"RC21 =   0.000000\n";
    ss<<"RC22 =   1.000000\n";
    ss<<"RC23 =   0.000000\n";
    ss<<"RC31 =   0.000000\n";
    ss<<"RC32 =   0.000000\n";
    ss<<"RC33 =   1.000000\n";


    ss<<"HEIGHT = 0.0000  # Height [m]\n";

    ss<<"RotMatInRowColumn = false   # Rotation matrix is transposed \n";
    ss<<"ReferenceFrameCurrentCam = true   # current cam is ref frame \n";

    ss<<"[GLRectification]\n";
    ss<<"UseLenseCorrection=false\n";
    ss<<"UseKKMatrix=false\n";

    ss<<"[WORLD]\n";
    ss<<"TC_X =   "<<baselinemm<<"  # Translation X [mm]\n";
    ss<<"TC_Y =   0.000000  # Translation Y [mm]\n";
    ss<<"TC_Z =   0.000000  # Translation Z [mm]\n";
    ss<<"RC11 =   1.000000  # Rotation matrix\n";
    ss<<"RC12 =   0.000000\n";
    ss<<"RC13 =   0.000000\n";
    ss<<"RC21 =   0.000000\n";
    ss<<"RC22 =   1.000000\n";
    ss<<"RC23 =   0.000000\n";
    ss<<"RC31 =   0.000000\n";
    ss<<"RC32 =   0.000000\n";
    ss<<"RC33 =   1.000000\n";

    ss<<"RotMatInRowColumn = false   # Rotation matrix is transposed \n";



    return ss.str();

}
void writeNgSequence(const Sequence& seq,std::string outputpath){
    outputpath=outputpath+seq.name();
    // check if the sequence
    // write the ngsystemconfig.xml
    {
        saferSystemCall("mkdir -p "+outputpath);

        std::stringstream ss;
        ss<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        ss<<"<ngserverlist>\n";
        ss<<"    <uvoserver idstring=\"Grabber\" type=\"23\" subtype=\"0\"/>\n";
        ss<<"</ngserverlist>";
        std::ofstream file;file.open(outputpath+"/ngsystemconfig.xml"); file<<ss.str()<<endl; file.close();

    }
    // write the config files
    {

        saferSystemCall("mkdir -p "+outputpath+"/systemData");
        {std::ofstream file;file.open(outputpath+"/systemData"+std::string("/calib_k0.bog")); file<<calibdata(seq.ks[0],seq.rows(),seq.cols())<<endl; file.close();}
        {std::ofstream file;file.open(outputpath+"/systemData"+std::string("/calib_k1.bog")); file<<calibdata(seq.ks[1],seq.rows(),seq.cols())<<endl; file.close();}
    }

    // write the time stamp files
    {
        std::string path=outputpath+"/Grabber_Dataset/";
        saferSystemCall("mkdir -p "+path);
        for(int i=0;i<seq.samples();++i){
            std::stringstream ss;
            ss<<"Timestamp: "<<seq.times()[i]<<"\n";
            ss<<"Framestamp: "<<i<<"\n";
            std::ofstream file; file.open(path+mlib::toZstring(i,5)+"_c0.txt");  file<<ss.str()<<endl;  file.close();
        }
    }

    // write the DatasetConfig.xml
    {
        std::string path=outputpath+"/Grabber_Dataset/DatasetConfig.xml";

        std::stringstream ss;
        ss<<"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        ss<<"<Grabber>\n";
        ss<<"  <ServerType>23</ServerType>\n";
        ss<<"  <NumCameras>2</NumCameras>\n";
        ss<<"  <Camera_0_Width>"<<seq.cols()<<"</Camera_0_Width>\n";
        ss<<"  <Camera_0_Height>"<<seq.rows()<<"</Camera_0_Height>\n";
        ss<<"  <Camera_0_Bits_per_pixel>12</Camera_0_Bits_per_pixel>\n";
        ss<<"  <Camera_0_Number_of_channels>1</Camera_0_Number_of_channels>\n";
        ss<<"  <Camera_1_Width>"<<seq.cols()<<"</Camera_1_Width>\n";
        ss<<"  <Camera_1_Height>"<<seq.rows()<<"</Camera_1_Height>\n";
        ss<<"  <Camera_1_Bits_per_pixel>12</Camera_1_Bits_per_pixel>\n";
        ss<<"  <Camera_1_Number_of_channels>1</Camera_1_Number_of_channels>\n";
        ss<<"</Grabber>\n";
        std::ofstream file; file.open(path);  file<<ss.str()<<endl;  file.close();

    }

    { // write the cameras.dat file to systemData
        std::stringstream ss;
        // this file is found under ip/ip_base/param
        ss<<"###############################################################################    \n";
        ss<<"#     Camera parameter file for ts_MonoCamera / ts_StereoCamera class.        #    \n";
        ss<<"###############################################################################    \n";
        ss<<"                                                                                   \n";
        ss<<"                                                                                   \n";
        ss<<"[INTERNAL]                                                                         \n";
        ss<<"F	=	0.004	   # [m] focal length                                               \n";
        ss<<"SX	=	7.92835e-6 # [m] pixel size in X direction                                  \n";
        ss<<"SY	=	7.85802e-6 # [m] pixel size in Y direction                                  \n";
        ss<<"X0	=	320	   # [pixel] X-coordinate of principle point                            \n";
        ss<<"Y0	=	240	   # [pixel] Y-coordinate of principle point                            \n";
        ss<<"                                                                                   \n";
        ss<<"                                                                                   \n";
        ss<<"[EXTERNAL]                                                                         \n";
        ss<<"B	=	0.3	   # [m] width of baseline of stereo camera rig                         \n";
        ss<<"LATPOS	=	0.0	   # [m] lateral position of camera                                 \n";
        ss<<"HEIGHT	=	1.21	   # [m] height of camera                                       \n";
        ss<<"DISTANCE=	0.0	   # [m] distance of camera                                         \n";
        ss<<"TILT	=	0.0	   # [rad] tilt angle                                               \n";
        ss<<"YAW	=	0.0	   # [rad] yaw angle                                                \n";
        ss<<"ROLL	= 	0.0	   # [rad] roll angle                                               \n";
        ss<<"                                                                                   \n";
        ss<<"                                                                                   \n";
        ss<<"                                                                                   \n";
        ss<<"# Notes:                                                                           \n";
        ss<<"#  In a stereo camera system the internal parameters for both cameras are the      \n";
        ss<<"#  same.                                                                           \n";
        ss<<"#                                                                                  \n";
        ss<<"#  The camera model is left handed. Looking in frontal direction, the x axis       \n";
        ss<<"#  shows to the right, the y axis is directed to the sky and the z axis shows      \n";
        ss<<"#  in driving direction.                                                           \n";
        ss<<"#  The world to camera transformation is performed by first a translation          \n";
        ss<<"#  (latpos, height, distance) followed by a rotation (tilt, yaw, roll).            \n";
        ss<<"#                                                                                  \n";
        ss<<"#  The angle directions are:                                                       \n";
        ss<<"#   tilt > 0  <=>  looking down                                                    \n";
        ss<<"#   yaw  > 0  <=>  looking right                                                   \n";
        ss<<"#   roll > 0  <=>  rolling clockwise                                               \n";
        ss<<"#                                                                                  \n";
        ss<<"# For more information see the inline documentation of ts_MonoCamera!              \n";
        std::string path=outputpath+"/systemData/cameras.dat";
        std::ofstream file; file.open(path);  file<<ss.str()<<endl;  file.close();
    }





    // write the images files

    std::string path=outputpath+"/Grabber_Dataset/";
    saferSystemCall("mkdir -p "+path);
    mlib::Timer timer;timer.tic();
    for(int i=0;i<seq.samples();++i){

        // if the image exists skip it... this relies on no incomplete file beeing written which is guarenteed by the rename and zfs transactions
        if(mlib::fileexists(path+mlib::toZstring(i,5)+"_c1.pgm",false)) continue;
        std::vector<cv::Mat1b> imgs;
        bool test=seq.getImages(imgs,i);
        if(!test){
            std::cout<<"Failed to read image"<<seq.sequence()<<":"<<i<<endl;
            exit(1);
        }

        assert(test && "all images must load !");

        // save as 8 bit pgm...
        // first write the files, then rename it in order to ensure its been completely written even if the program is aborted
        cv::Mat1w im16=toMat_<unsigned short, unsigned char>(imgs[0]);            im16*=16;
        cv::imwrite( path+toZstring(i,5)+"_c0_bak.pgm",im16);
        std::rename((path+toZstring(i,5)+"_c0_bak.pgm").c_str(),
                    (path+toZstring(i,5)+"_c0.pgm").c_str());
        im16=toMat_<unsigned short, unsigned char>(imgs[1]);            im16*=16;
        cv::imwrite( path+toZstring(i,5)+"_c1_bak.pgm",im16);
        std::rename((path+toZstring(i,5)+"_c1_bak.pgm").c_str(),
                    (path+toZstring(i,5)+"_c1.pgm").c_str());



        if(i % 100 == 0){
            // write log file

            timer.toc();
            cout<<"img: "<<i<<" of "<<seq.samples()<<" of sequence "<<seq.name() <<" time: "<<timer.median()<<endl;
            timer.tic();
        }


    }

    cout<<"wrote "<<outputpath<<endl;

}




}// end kitti namespace
}// end namespace cvl
