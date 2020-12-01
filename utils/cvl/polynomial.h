#pragma once
#include <iostream>
#include <sstream>
#include <array>
#include <mlib/utils/cvl/matrix.h>
#include <vector>
#include <map>

namespace cvl{
template<unsigned int degree>
class Polynomial
{
public:

    // 1, x, x², x³ osv...
    Vector<double,degree+1> coeffs=Vector<double,degree+1>::Zero();
    Polynomial(){}

    template<unsigned int O>
    Polynomial(Polynomial<O> o){
        //static_assert(O<=degree, "");
        for(uint i=0;i<std::min(o.coeffs.size(),coeffs.size());++i)
            coeffs[i]=o.coeffs[i];
    }


    template<class... V>
    Polynomial(double v0, V... v){
        static_assert(sizeof...(V)<=degree, "");
        std::array<double,sizeof...(V)+1> arr{v0, double(v)...};
        for(uint i=0;i<coeffs.size();++i)
            coeffs[i]=0;
        for(uint i=0;i<arr.size();++i)
            coeffs[i]=arr[i];
    }

    std::string str(){
        std::stringstream ss;
        ss<<"p(x) = ";

        auto rev=coeffs.reverse();
        // coeffs are in x5, x4,x3 etc degree
        for(uint i=0;i<coeffs.size();++i)
        {
            if(rev[i]==0.0) continue;
            ss<<rev[i];
            if(i!=coeffs.size()-1)
                ss<<"x";
            if(coeffs.size()-i>2)
                ss<<"^"<<coeffs.size()-i-1;
            if(i!=coeffs.size()-1)
                ss<<" + ";
        }
        return ss.str();
    }

    Polynomial<std::max(0,int(degree)-1)> derivative() const{
        Polynomial<std::max(0,int(degree)-1)> poly;

        for(uint i=1;i<coeffs.size();++i)
        {
            poly.coeffs[i-1] = coeffs[i]*i;
        }
        return poly;
    }

    Polynomial<degree+1> primitive()
    {

        Polynomial<degree+1> poly;
        poly.coeffs[0]=0; // or some value who knows
        for(uint i=0;i<coeffs.size();++i)
        {

            poly.coeffs[i+1] = coeffs[i]/double(i+1);
        }
        return poly;
    }


    template<unsigned int Inc> Polynomial<degree+Inc> inc_degree(){

        Polynomial<degree+Inc> poly;
        for(uint i=0;i<coeffs.size();++i)
        {
            poly.coeffs[i] = coeffs[i];
        }
        return poly;
    }
    double operator()(double x){
        double v=coeffs[0];
        for(uint i=1;i<coeffs.size();++i){
            v+=coeffs[i]*std::pow(x,double(i));
        }
        return v;
    }

    template<int P>
    constexpr Polynomial<P*degree> pow_int() const{
        // slow
        static_assert(P>1, "does only work for higher pows... ");

        if constexpr (P==2){
            return (*this)*(*this);
        }
        else return (*this)*pow_int<P-1>();
    }

    template<unsigned int out>
    Polynomial<out> pow_int2(int P) const{


        if(P==0) return Polynomial<out>(1);
        if(P==1) return Polynomial<out>(*this);
        if(P==2) return Polynomial<out>((*this)*(*this));

        return Polynomial<out>((*this)*pow_int2<out>(P-1));
    }

    Polynomial<20> pow_int22(int P) const{


        if(P==0) return Polynomial<20>(1);
        if(P==1) return Polynomial<20>(*this);
        if(P==2) return Polynomial<20>((*this)*(*this));

        return Polynomial<20>((*this)*pow_int2<20>(P-1));
    }


    // t=x+d;
    Polynomial<degree> reparam(double d){
        Polynomial<degree> out;
        Polynomial<1> diff(d,1);


        for(uint i=0;i<coeffs.size();++i)
        {
            out=out + coeffs[i]*diff.pow_int2<degree>(i);
        }
        return out;
    }
    bool zero(){
        for(auto a:coeffs)
            if(a!=0) return false;
        return true;
    }
};

template<unsigned int A,unsigned  int B>
bool operator==(Polynomial<A> a, Polynomial<B> b){
    Polynomial<std::max(A,B)> c=a-b;
    for(double d:c.coeffs){
        if(std::abs(d)>=1e-15) return false;
    }
    return true;
}
template<unsigned int A,unsigned  int B>
bool operator!=(Polynomial<A> a, Polynomial<B> b){return !(a==b);}



template<unsigned int A, unsigned int B>
Polynomial<std::max(A,B)> operator+(Polynomial<A> a,
               Polynomial<B> b){
    Polynomial<std::max(A,B)> poly;
    for(uint i=0;i<poly.coeffs.size();++i){

        poly.coeffs[i]=0.0;
        if(i<a.coeffs.size())
            poly.coeffs[i]+=a.coeffs[i];
        if(i<b.coeffs.size())
            poly.coeffs[i]+=b.coeffs[i];
    }
    return poly;
}

template<unsigned int A, unsigned int B>
Polynomial<std::max(A,B)> operator-(Polynomial<A> a,
               Polynomial<B> b){
    Polynomial<std::max(A,B)> poly;
    for(uint i=0;i<poly.coeffs.size();++i){

        poly.coeffs[i]=0.0;
        if(i<a.coeffs.size())
            poly.coeffs[i]+=a.coeffs[i];
        if(i<b.coeffs.size())
            poly.coeffs[i]-=b.coeffs[i];
    }
    return poly;
}

template<unsigned int A, unsigned int B>
Polynomial<A+B> operator*(Polynomial<A> a,
                          Polynomial<B> b){
    Polynomial<A+B> poly;
    for(uint i=0;i<a.coeffs.size();++i)
        for(uint j=0;j<b.coeffs.size();++j)
            poly.coeffs[i+j] += a.coeffs[i]*b.coeffs[j];
    return poly;
}



template<unsigned int A, class T> auto operator+(Polynomial<A> a, T t){
    return a+Polynomial<0>(t);
}
template<unsigned int A, class T> auto operator-(Polynomial<A> a, T t){
    return a-Polynomial<0>(t);
}
template<unsigned int A, class T> auto operator*(Polynomial<A> a, T t){
    return a*Polynomial<0>(t);
}

template<unsigned int A, class T> auto operator+(T t,Polynomial<A> a){
    return a+Polynomial<0>(t);
}
template<unsigned int A, class T> auto operator-(T t,Polynomial<A> a){
    return a-Polynomial<0>(t);
}
template<unsigned int A, class T> auto operator*(T t,Polynomial<A> a){
    return a*Polynomial<0>(t);
}



template<unsigned int S>
std::ostream& operator<<(std::ostream& os, Polynomial<S> poly){
    return os<<poly.str();
}

template<unsigned int degree>
class BoundedPolynomial
{
public:
    Vector2d bounds; // only nonzero inside!

    Polynomial<degree> p;

    template<class... V>
    BoundedPolynomial(Vector2d bounds, V... v):bounds(bounds),p(v...){}

    template<unsigned int O>
    BoundedPolynomial(Vector2d bounds, Polynomial<O> o):bounds(bounds),p(o){}

    template<unsigned int O>
    BoundedPolynomial(BoundedPolynomial<O> o):bounds(o.bounds),p(o.p){}
    BoundedPolynomial(){}

    bool good_bounds()
    {
        return(bounds[0]<bounds[1]);
    }
    bool zero(){
        return p.zero();
    }
    // t=x+d
    BoundedPolynomial reparam(double d){
       return  BoundedPolynomial(bounds - Vector2d(d,d),p.reparam(d));
    }


    template<unsigned int A> Vector2d overlap(BoundedPolynomial<A> bp){
        if(!good_bounds()) return Vector2d(0,0);
        if(!bp.good_bounds()) return Vector2d(0,0);
        return Vector2d(std::max(bounds[0],bp.bounds[0]),
                std::min(bounds[1],bp.bounds[1]));
    }



    std::string str(){
        std::stringstream ss;
        ss<<"bounds: "<<bounds<<" ";
        ss<<p;
        return ss.str();
    }


    auto unbounded(){
        return p;
    }

    auto derivative() const
    {
        // not correct, ignored boundries
        return BoundedPolynomial<degree>(bounds,p.derivative());
    }
    std::vector<BoundedPolynomial<degree+1>> primitive(){
        std::vector<BoundedPolynomial<degree+1>> ret;
        auto P=p.primitive();
        ret.push_back(BoundedPolynomial(bounds,P));
        //ret.push_back(BoundedPolynomial({bounds,std::numeric_limits<double>::max()},Polynomial))
    }



    template<class T>
    T operator()(T x){
        if(x<bounds[0]) return T(0);
        if(!(x<bounds[1])) return T(0);
        return p(x);
    }

    template<unsigned int out>
    BoundedPolynomial<out> pow_int2(int P) const
    {
        return BoundedPolynomial<out>(bounds,p.pow_int22(P));
    }


};




template<unsigned int A, unsigned int B>
BoundedPolynomial<A+B> operator*(BoundedPolynomial<A> a,
               BoundedPolynomial<B> b){


    if(a.overlap(b).norm()>0)
        return BoundedPolynomial<A+B>(a.overlap(b), Polynomial<A+B>(a.p*b.p));
    return BoundedPolynomial<A+B>(a.overlap(b),   Polynomial<A+B>());
}

template<unsigned int A> BoundedPolynomial<A> operator*(BoundedPolynomial<A> a, double t){
    return a*BoundedPolynomial<0>(a.bounds,t);
}


template<unsigned int A> BoundedPolynomial<A> operator*(double t,BoundedPolynomial<A> a){
    return a*BoundedPolynomial<0>(a.bounds,t);
}

template<unsigned int S>
std::ostream& operator<<(std::ostream& os, BoundedPolynomial<S> poly){
    return os<<poly.str();
}


template<class T> bool less(const Vector2<T>& a,const Vector2<T>& b){
    for(int i=0;i<2;++i){
    if(a[i]<b[i]) return true;
    if(a[i]>b[i]) return false;
    }
    return false;
};


template<unsigned int degree>
class CompoundBoundedPolynomial
{
public:
    std::vector<BoundedPolynomial<degree>> polys;
    template<class T>
    T operator()(T x){
        T v=0;
        for(auto p:polys)
            v+=p(x);
    }

    CompoundBoundedPolynomial reparam(double d){
        CompoundBoundedPolynomial cbp;
        for(auto poly:polys)
            cbp.add(poly.reparam(d));
        return cbp;
    }

    void add(BoundedPolynomial<degree> p){
        if(p.zero()) return;
        if(p.good_bounds())
            polys.push_back(p);
    }
    std::vector<double> integrate(){
        std::vector<double> ds;
        for(auto poly:polys){
            ds.push_back(poly.integrate());
        }
        return ds;
    }
    std::string str(){
        std::stringstream ss;
        for(auto p:polys)
            ss<<p<<"\n";
        return ss.str();
    }
    Vector2d span(){
        if(polys.size()==0) return {0,0};
        double low=polys[0].bounds[0];
        double high=polys[0].bounds[1];
        for(auto p:polys){
            if(p.bounds[0]<low) low=p.bounds[0];
            if(p.bounds[1]>high) high=p.bounds[1];
        }
        return {low,high};
    }

    CompoundBoundedPolynomial<degree> collapse(){


        std::map<std::array<double,2>, Polynomial<degree>> map;
        for(auto p:polys){
            // auto inserts zero
            map[p.bounds.std_array()] = map[p.bounds.std_array()] + p.p;
        }
        CompoundBoundedPolynomial<degree> cbp;
        for(auto [b,p]:map){
            cbp.add(BoundedPolynomial<degree>(Vector2d(b[0],b[1]),p));
        }
        return cbp;
    }

};

template<unsigned int A, unsigned int B>
CompoundBoundedPolynomial<A+B>
operator*(CompoundBoundedPolynomial<A> a,
          CompoundBoundedPolynomial<B> b){
    CompoundBoundedPolynomial<A+B> out;
    for(auto ap:a.polys){
        for(auto bp:b.polys){
            out.add(ap*bp);
        }
    }
    return out;
}

template<unsigned int A, unsigned int B>
CompoundBoundedPolynomial<A+B>
operator*(CompoundBoundedPolynomial<A> a,
          Polynomial<B> b)
{
    CompoundBoundedPolynomial<B> c;c.add(BoundedPolynomial<B>(a.span(),b));
    return a*c;
}

template<unsigned int A, unsigned int B>
CompoundBoundedPolynomial<A+B>
operator+(CompoundBoundedPolynomial<A> a,
          CompoundBoundedPolynomial<B> b){
    CompoundBoundedPolynomial<A+B> out;
    for(auto ap:a.polys){
        for(auto bp:b.polys){
            auto span=ap.overlap(bp);
            if(span.norm()>0)
                out.add(BoundedPolynomial(span,ap.p +bp.p));
        }
    }
    return out;
}
template<unsigned int A, unsigned int B>
CompoundBoundedPolynomial<A+B>
operator-(CompoundBoundedPolynomial<A> a,
          CompoundBoundedPolynomial<B> b){
    CompoundBoundedPolynomial<A+B> out;
    for(auto ap:a.polys){
        for(auto bp:b.polys){
            auto span=ap.overlap(bp);
            if(span.norm()>0)
                out.add(BoundedPolynomial(span,ap.p -bp.p));
        }
    }
    return out;
}




template<unsigned int S>
std::ostream& operator<<(std::ostream& os, CompoundBoundedPolynomial<S> poly){
    return os<<poly.str();
}

}
