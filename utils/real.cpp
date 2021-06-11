#include <mlib/utils/real.h>
#include <iomanip>
#include <cmath>


namespace cvl {






 Real sqrt(Real real){

    Real tmp=real;
    tmp.f=std::sqrt(tmp.f);
    tmp.d=std::sqrt(tmp.d);
    tmp.l=std::sqrt(tmp.l);
   return tmp;
}

 Real pow(Real real,Real to){

    Real tmp=real;
    tmp.f=std::pow(tmp.f,to.f);
    tmp.d=std::pow(tmp.d,to.d);
    tmp.l=std::pow(tmp.l,to.l);
   return tmp;
}
 Real exp(Real real){

    Real tmp=real;
    tmp.f=std::exp(tmp.f);
    tmp.d=std::exp(tmp.d);
    tmp.l=std::exp(tmp.l);
   return tmp;
}
 Real sin(Real real){

    Real tmp=real;
    tmp.f=std::sin(tmp.f);
    tmp.d=std::sin(tmp.d);
    tmp.l=std::sin(tmp.l);
   return tmp;
}
 Real cos(Real real){

    Real tmp=real;
    tmp.f=std::cos(tmp.f);
    tmp.d=std::cos(tmp.d);
    tmp.l=std::cos(tmp.l);
   return tmp;
}
 Real log(Real real){

    Real tmp=real;
    tmp.f=std::log(tmp.f);
    tmp.d=std::log(tmp.d);
    tmp.l=std::log(tmp.l);
   return tmp;
}
 Real atan2(Real a,Real b){
   return Real(std::atan2(a.f,b.f),std::atan2(a.d,b.d),std::atan2(a.l,b.l));
}

 std::ostream& operator<<(std::ostream &os, const Real& t){
    os<<std::setprecision (30)<<t.f<<" "<<t.d<<" "<<t.l<<" diff: "<<(long double)t.f-t.l<<" "<<(long double)t.d-t.l<<std::endl;
    return os;
}
}
