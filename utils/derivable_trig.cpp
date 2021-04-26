#include <cmath>
#include <mlib/utils/derivable_trig.h>
namespace cvl{
double dcos(double w, double t, uint derivative){
    if(derivative==0) return std::cos(w*t);
    return -w*dsin(w,t,derivative-1);
    //double wn=std::pow(w,derivative);
    //if(derivative % 2 ==0) return wn*std::cos(w*t);
    //return -wn*std::sin(w*t);
}
double dsin(double w, double t, uint derivative){
    if(derivative==0) return std::sin(w*t);
    return w*dcos(w,t,derivative-1);
}
}
