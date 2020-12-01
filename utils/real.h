#pragma once
/* ********************************* FILE ************************************/
/** \file    real.h
 *
 * \brief    This header contains the real class which tracks where floating point numerical errors occur
 *
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - tested by test_real.cpp
 * - header only
 *
 * Everything operates on the notion that the long doubles are the true values. Thus the path is selected by them
 * but any incongruent behaviour in the float or double is visible!
 * 
 *
 *
 *
 * \todo
 * - write test
 *
 * \author   Mikael Persson
 * \date     2014-04-01
 * \note MIT licence
 *
 ******************************************************************************/




#include <iostream>




namespace cvl{



class Real{
public:
    float f;
    double d;
    long double l;
    // add int too?

    Real():f(0),d(0),l(0){}
    Real(float f, double d, long double l):f(f),d(d),l(l){}    
    //Real(long double t): f(float(t)),d(double(t)),l(t){}

    Real operator-() const{return Real(-f,-d,-l);}
    Real operator+(Real b) const{return Real(f+b.f,d+b.d,l+b.l);}
    Real operator-( Real b) const{return Real(f-b.f,d-b.d,l-b.l);}
    Real operator*( Real b) const{return Real(f*b.f,d*b.d,l*b.l);}
    Real operator/( Real b) const{return Real(f/b.f,d/b.d,l/b.l);}
    //Real operator%( Real b) const{return Real(f % b.f, d % b.d, l % b.l);}

    Real& operator+=(Real b )    {        f+=b.f;       d+=b.d;        l+=b.l;        return *this;    }
    Real& operator-=(Real b )    {        f-=b.f;        d-=b.d;        l-=b.l;        return *this;    }
    Real& operator*=(const Real& b )    {        f*=b.f;        d*=b.d;        l*=b.l;        return *this;    }
    Real& operator/=(const Real& b )    {        f/=b.f;        d/=b.d;        l/=b.l;        return *this;    }

    // thresholds! operate on the highest precision, in class lets me mark divergence...
    bool operator==(const Real rhs) const{        return l == rhs.l;   }
    bool operator< (const Real rhs) const{        return l < rhs.l;    }
    // consequences
    bool operator!=(const Real rhs) const{return l!=rhs.l;}
    bool operator> (const Real rhs) const{return l>rhs.l;}
    bool operator<=(const Real rhs) const{return l<=rhs.l;}
    bool operator>=(const Real rhs) const{return l>=rhs.l;}


};




Real sqrt(Real real);
Real pow(Real real,Real to);
 Real exp(Real real);
 Real sin(Real real);
 Real cos(Real real);
 Real log(Real real);
 Real atan2(Real a,Real b);

 std::ostream& operator<<(std::ostream &os, const Real& t);

}// end namespace cvl
