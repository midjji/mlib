#if 1
#include <string>
#include <mlib/utils/string_helpers.h>
#include <fstream>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/files.h>
#include <mlib/plotter/plot.h>
#include <mlib/utils/mlibtime.h>
using namespace mlib;
using namespace cvl;
using std::cout;
using std::endl;
double read(std::string pth){

    /*
lat:   latitude of the oxts-unit (deg)
lon:   longitude of the oxts-unit (deg)
alt:   altitude of the oxts-unit (m)
roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
vn:    velocity towards north (m/s)
ve:    velocity towards east (m/s)
vf:    forward velocity, i.e. parallel to earth-surface (m/s)
vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
az:    acceleration in z, i.e. in direction of vehicle top (m/s^2)
af:    forward acceleration (m/s^2)
al:    leftward acceleration (m/s^2)
au:    upward acceleration (m/s^2)
wx:    angular rate around x (rad/s)
wy:    angular rate around y (rad/s)
wz:    angular rate around z (rad/s)
wf:    angular rate around forward axis (rad/s)
wl:    angular rate around leftward axis (rad/s)
wu:    angular rate around upward axis (rad/s)
pos_accuracy:  velocity accuracy (north/east in m)
vel_accuracy:  velocity accuracy (north/east in m/s)
navstat:       navigation status (see navstat_to_string)
numsats:       number of satellites tracked by primary GPS receiver
posmode:       position mode of primary GPS receiver (see gps_mode_to_string)
velmode:       velocity mode of primary GPS receiver (see gps_mode_to_string)
orimode:       orientation mode of primary GPS receiver (see gps_mode_to_string)
*/

            std::ifstream ifs(pth);

            for(int i=0;i<17;++i){
                double d=0;
                ifs>>d;

            }
            Vector3d av;
            for(int i=0;i<3;++i){
                ifs>>av[i];


            }

            return av.norm();
}



int main()
{
    std::string path="/home/mikael/tmp2/2011_09_26/2011_09_26_drive_0046_sync/oxts/data/";
         std::vector<double> omegas;omegas.reserve(10000);
         std::vector<double> ts;

    for(int i=0;i<1000;++i){
        std::string fp=path+mlib::toZstring(i,10)+".txt";
        if(!fileexists(fp,false)) break;




ts.push_back(i*0.1);
   omegas.push_back(read(fp));

    }
    std::sort(omegas.begin(), omegas.end());
    for(auto d:omegas)
        cout<<"d: "<<d<<std::endl;
initialize_plotter();
    plot(ts,omegas,"omegas", "0001");

    mlib::sleep(1000);
    return 0;
}

#else

#include "log.h"
#include <iostream>
#include <thread>
using std::cout;using std::endl;

using namespace cvl;
void function_a(){
    mlog()<<"msg in function a"<<endl;
}
void function_b(){
    mlog()<<"msg in function b"<<endl;
}
class AClass{
public:
    AClass(){}
    void fun(){
        mlog()<<"msg in AClass::fun"<<endl;
    }
};

class BClass{
public:
    BClass(){}
    void afun(){
        mlog()<<"msg in AClass::fun"<<endl<<"\n"<<"tstse "<<"\nmore...";
    }
};
void test(){
    mlog().set_thread_name("test thread");
    AClass b;
    b.fun();
}




int main(){






    while(true){
        if(true){

            mlog()<<endl;
            mlog()<<"mlog here"<<endl;



            mlog()<<"test 1"<<" and more test1"<< " and even more test 1"<<endl;


            function_a();

            function_b();
            AClass as;
            as.fun();

            std::thread thr(test);

            thr.join();

            printf(mlog(),"djaksdjflasdjflasdköfj %05d.txt\0\n",10);
        }

        break;
    }




    return 0;
}
#endif
