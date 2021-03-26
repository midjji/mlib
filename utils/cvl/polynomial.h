#pragma once
#include <iostream>
#include <sstream>
#include <array>
#include <vector>
#include <map>
#include <algorithm>

#include <mlib/utils/cvl/matrix.h>


namespace cvl {



template<unsigned int degree, class Type=long double>
class Polynomial {
public:

    // 1, x, x², x³ osv...
    Vector<Type,degree+1> coeffs=Vector<Type,degree+1>::Zero();
    Polynomial(){}
    Polynomial(long double v0){coeffs[0]=Type(v0);}

    template<unsigned int O>
    Polynomial(Polynomial<O,Type> o){
        //static_assert(O<=degree, "");
        for(uint i=0;i<std::min(o.coeffs.size(),coeffs.size());++i)
            coeffs[i]=o.coeffs[i];
    }

    void zero_eps_coefficients(){
        if constexpr(std::is_floating_point<Type>()){
            for(auto& c:coeffs)
                if(std::abs(c)<1e-15 )c=0;
        }
    }


    template<class... V>
    Polynomial(Type v0, V... v){
        static_assert(sizeof...(V)<=degree, "");
        std::array<Type,sizeof...(V)+1> arr{v0, Type(v)...};
        for(uint i=0;i<coeffs.size();++i)
            coeffs[i]=Type(0);
        for(uint i=0;i<arr.size();++i)
            coeffs[i]=arr[i];
    }

    std::string str(){


        auto rev=coeffs.reverse();

        std::vector<std::string> parts;

        // coeffs are in x5, x4,x3 etc degree
        for(uint i=0;i<coeffs.size();++i)
        {
            std::stringstream ss;
            if(rev[i]==0.0) continue;
            //if(i!=coeffs.size()-1 && !(rev[i]==1.0))
            ss<<rev[i];

            if(i!=coeffs.size()-1)
                ss<<"x";
            if(coeffs.size()-i>2)
                ss<<"^"<<coeffs.size()-i-1;
            parts.push_back(ss.str());
        }
        if(parts.size()==0) parts.push_back("0");
        std::stringstream ss;
        ss<<"p(x) = ";
        for(uint i=0;i<parts.size();++i){
            ss<<parts[i];
            if(i!=parts.size()-1)
                ss<<" + ";
        }
        return ss.str();
    }

    std::string str2(){
        std::stringstream ss;
        ss<<coeffs;
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

            poly.coeffs[i+1] = coeffs[i]/Type(i+1);
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
    long double operator()(Type x) const{
        long double v=coeffs[0];
        for(uint i=1;i<coeffs.size();++i){
            v+=coeffs[i]*std::pow(x,Type(i));
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
    Polynomial<out,Type> pow_int2(int P) const{


        if(P==0) return Polynomial<out,Type>(1);


        if(P==1) return Polynomial<out,Type>(*this);


        if(P==2) return Polynomial<out,Type>((*this)*(*this));


        return Polynomial<out,Type>((*this)*pow_int2<out>(P-1));
    }
    template<unsigned int exp>
    Polynomial<exp*degree,Type> pow() const
    {
        if constexpr(exp==0) return Polynomial<0,Type>(1);
        if constexpr(exp==1) return *this;
        else
        return (*this)*pow<exp-1>();
    }




    // x=x-1
    Polynomial<degree,Type> reparam(long double d){
        Polynomial<20,Type> out;
        Polynomial<20,long double> diff(d,1);


        for(uint i=0;i<coeffs.size();++i)
        {
            Polynomial<20,Type> tmp=diff.pow_int2<20>(i)*Polynomial<0,Type>(coeffs[i]);
            //cout<<"reparam1: "<<diff.pow_int2<20>(i).str()<<": i is "<<i<<endl;
            out=out + tmp;
        }
        //cout<<"reparam0: "<<diff.str()<<endl;
        //cout<<"reparam: "<<out<<endl;
        return Polynomial<degree,Type>(out);
    }
    bool zero(){
        for(auto a:coeffs)
            if(!(a==0)) return false;
        return true;
    }
};

template<unsigned int A,unsigned  int B,class Type>
bool operator==(Polynomial<A,Type> a, Polynomial<B,Type> b){
    for(uint i=std::min(A,B);i<std::max(A,B);++i){
        if(A>B){
            if(!(a.coeffs[i]==0)) return false;
        }
        if(B>A){
            if(!(b.coeffs[i]==0)) return false;
        }
    }
    for(uint i=0;i<std::min(A,B);++i){
        if(!(a.coeffs[i]==b.coeffs[i]))return false;
    }
    return true;
}
template<unsigned int A,unsigned  int B>
bool operator!=(Polynomial<A> a, Polynomial<B> b){return !(a==b);}



template<unsigned int A, unsigned int B, class Type>
Polynomial<std::max(A,B),Type> operator+(Polynomial<A,Type> a,
                                         Polynomial<B,Type> b){
    Polynomial<std::max(A,B),Type> poly;
    for(uint i=0;i<poly.coeffs.size();++i){

        poly.coeffs[i]=0.0;
        if(i<a.coeffs.size())
            poly.coeffs[i]+=a.coeffs[i];
        if(i<b.coeffs.size())
            poly.coeffs[i]+=b.coeffs[i];
    }
    poly.zero_eps_coefficients();
    return poly;
}

template<unsigned int A, unsigned int B, class Type>
Polynomial<std::max(A,B)> operator-(Polynomial<A,Type> a,
                                    Polynomial<B,Type> b){
    Polynomial<std::max(A,B),Type> poly;
    for(uint i=0;i<poly.coeffs.size();++i){

        poly.coeffs[i]=0.0;
        if(i<a.coeffs.size())
            poly.coeffs[i]+=a.coeffs[i];
        if(i<b.coeffs.size())
            poly.coeffs[i]-=b.coeffs[i];
    }
    return poly;
}






template<unsigned int A, unsigned int B, class Type1, class Type2>
auto operator*(Polynomial<A,Type1> a,
               Polynomial<B,Type2> b){
    //cout<<"poly: "<<a<<" b: "<<b<<endl;
    using T=decltype(Type1(0) * Type2(0));
    Polynomial<A+B,T> poly;
    for(uint i=0;i<a.coeffs.size();++i)
        for(uint j=0;j<b.coeffs.size();++j){

            poly.coeffs[i+j] =poly.coeffs[i+j] + a.coeffs[i]*b.coeffs[j];
        }

    return poly;
}


/*
template<unsigned int A>
Polynomial<A,Symb> operator*(Polynomial<A,long double> a,
                             Symb b){

    Polynomial<A,Symb> poly;
    for(uint i=0;i<a.coeffs.size();++i)
        poly.coeffs[i] = b*a.coeffs[i];

    return poly;
}*/


template<unsigned int A, class T, class T1> auto operator+(Polynomial<A,T> a, T1 t){
    return a+Polynomial<0,T1>(t);
}
template<unsigned int A, class T, class T1> auto operator-(Polynomial<A,T> a, T1 t){
    return a-Polynomial<0,T1>(t);
}
template<unsigned int A, class T, class T1> auto operator*(Polynomial<A,T> a, T1 t){
    return a*Polynomial<0,T1>(t);
}

template<unsigned int A, class T, class T1> auto operator+(T1 t,Polynomial<A,T> a){
    return a+Polynomial<0,T1>(t);
}
template<unsigned int A, class T, class T1> auto operator-(T1 t,Polynomial<A,T> a){
    return a-Polynomial<0,T1>(t);
}
template<unsigned int A, class T, class T1> auto operator*(T1 t,Polynomial<A,T> a){
    return a*Polynomial<0,T1>(t);
}


template<unsigned int S, class Type>
std::ostream& operator<<(std::ostream& os, Polynomial<S,Type> poly){
    return os<<poly.str();
}
template<unsigned int degree, class Type=long double>
class BoundedPolynomial
{
public:
    Vector2<long double> bounds; // only nonzero inside!

    Polynomial<degree,Type> p;

    template<class... V>
    BoundedPolynomial(Vector2<long double> bounds, V... v):bounds(bounds),p(v...){}

    template<unsigned int O>
    BoundedPolynomial(Vector2<long double> bounds, Polynomial<O,Type> o):bounds(bounds),p(o){}

    template<unsigned int O>
    BoundedPolynomial(BoundedPolynomial<O,Type> o):bounds(o.bounds),p(o.p){}
    BoundedPolynomial(){}

    bool good_bounds()
    {
        return(bounds[0]<bounds[1]);
    }

    void bound(Vector2<long double> b)
    {
        bounds=Vector2<long double>(std::max(bounds[0],b[0]),std::min(bounds[1],b[1]));
    }
    bool zero(){
        return p.zero();
    }
    // t=x+d
    BoundedPolynomial reparam(long double d){
        return  BoundedPolynomial(bounds - Vector2<long double>(d,d),p.reparam(d));
    }


    template<unsigned int A> Vector2<long double> overlap(BoundedPolynomial<A,Type> bp){
        if(!good_bounds()) return Vector2<long double>(0,0);
        if(!bp.good_bounds()) return Vector2<long double>(0,0);
        return Vector2<long double>(std::max(bounds[0],bp.bounds[0]),
                std::min(bounds[1],bp.bounds[1]));
    }



    std::string str(){
        std::stringstream ss;
        ss<<"bounds: ["<<bounds[0]<<","<<bounds[1]<<") ";
        ss<<p.str();
        return ss.str();
    }


    std::string str2(){
        std::stringstream ss;
        ss<<"bounds: ["<<bounds[0]<<","<<bounds[1]<<") ";
        ss<<p.str2();
        return ss.str();
    }

    auto unbounded(){
        return p;
    }

    auto derivative() const
    {
        // not correct, ignored boundries
        return BoundedPolynomial<degree,Type>(bounds,p.derivative());
    }
    std::vector<BoundedPolynomial<degree+1,Type>> primitive(){
        std::vector<BoundedPolynomial<degree+1,Type>> ret;
        auto P=p.primitive();
        ret.push_back(BoundedPolynomial(bounds,P));
        //ret.push_back(BoundedPolynomial({bounds,std::numeric_limits<long double>::max()},Polynomial))
    }



    template<class T>
    T operator()(T x) const{
        if(x<bounds[0]) return T(0);
        if(!(x<bounds[1])) return T(0);
        return p(x);
    }

    template<unsigned int exp>
    auto pow() const
    {

        return BoundedPolynomial(bounds,p. template pow<exp>());
    }



};




template<unsigned int A, unsigned int B, class Type>
BoundedPolynomial<A+B,Type> operator*(BoundedPolynomial<A,Type> a,
                                      BoundedPolynomial<B,Type> b){


    if(a.overlap(b).norm()>0)
        return BoundedPolynomial<A+B,Type>(a.overlap(b), Polynomial<A+B,Type>(a.p*b.p));
    return BoundedPolynomial<A+B,Type>(a.overlap(b),   Polynomial<A+B,Type>());
}

template<unsigned int A,class Type> BoundedPolynomial<A,Type> operator*(BoundedPolynomial<A,Type> a, long double t){
    return a*BoundedPolynomial<0,Type>(a.bounds,t);
}


template<unsigned int A> BoundedPolynomial<A> operator*(long double t,BoundedPolynomial<A> a){
    return a*BoundedPolynomial<0>(a.bounds,t);
}




template<class T> bool less(const Vector2<T>& a,const Vector2<T>& b){
    for(int i=0;i<2;++i){
        if(a[i]<b[i]) return true;
        if(a[i]>b[i]) return false;
    }
    return false;
};


template<unsigned int S,class Type>
std::ostream& operator<<(std::ostream& os, BoundedPolynomial<S,Type> poly){
    return os<<poly.str();
}
template<unsigned int degree,class Type=long double>
class CompoundBoundedPolynomial
{
public:
    std::vector<BoundedPolynomial<degree,Type>> polys;
    template<class T>
    T operator()(T x) const{
        T v=0;
        for(const auto& p:polys)
            v+=p(x);
        return v;
    }

    CompoundBoundedPolynomial reparam(long double d){
        CompoundBoundedPolynomial cbp;
        for(auto poly:polys)
            cbp.add(poly.reparam(d));
        return cbp;
    }

    void add(BoundedPolynomial<degree,Type> p){
        if(p.zero()) return;
        if(p.good_bounds())
            polys.push_back(p);
    }
    std::vector<long double> integrate(){
        std::vector<long double> ds;
        for(auto poly:polys){
            ds.push_back(poly.integrate());
        }
        return ds;
    }
    CompoundBoundedPolynomial derivative() const{
        CompoundBoundedPolynomial ret;
        for(auto p:polys)
            ret.add(p.derivative());
        return ret;
    }
    CompoundBoundedPolynomial derivative(uint N) const{

        if(N>degree){
            CompoundBoundedPolynomial ret;
            return ret;
        }
        if(N==0) return *this;
        if(N==1){
            return derivative();
        }
        return derivative(N-1).derivative();
    }
    std::string str(){
        std::stringstream ss;
        for(auto p:polys)
            ss<<p.str()<<"\n";
        return ss.str();
    }
    std::string str2(){
        std::stringstream ss;
        for(auto p:polys)
            ss<<p.str2()<<"\n";
        return ss.str();
    }
    Vector2<long double> span(){
        if(polys.size()==0) return {0,0};
        long double low=polys[0].bounds[0];
        long double high=polys[0].bounds[1];
        for(auto p:polys){
            if(p.bounds[0]<low) low=p.bounds[0];
            if(p.bounds[1]>high) high=p.bounds[1];
        }
        return {low,high};
    }

    void collapse(){
        // first remove any polys with invalid bounds,
        std::vector<BoundedPolynomial<degree,Type>> ps;
        for(auto p:polys)
            if(p.good_bounds())
                ps.push_back(p);


        std::map<std::array<long double,2>, Polynomial<degree,Type>> map;
        for(auto p:ps)
        {
            // auto inserts zero
            map[p.bounds.std_array()] = map[p.bounds.std_array()] + p.p;
        }

        polys.clear();
        for(auto [b,p]:map){
            add(BoundedPolynomial<degree,Type>(Vector2<long double>(b[0],b[1]),p));
        }

    }

    Vector<long double,degree+1> compute_bcoeffs(long double t){
        Vector<long double,degree+1> x;
        for(uint i=0;i<x.rows();++i)
            x[i]=reparam(i)(t);
        return x;
    }
    CompoundBoundedPolynomial& operator*=(long double d){
        for(auto& p:polys)
            p.p.coeffs*=Type(d);
        return *this;
    }
    CompoundBoundedPolynomial& operator+=(CompoundBoundedPolynomial b){
        for(auto p: b.polys)
            add(p);
    }
    void bound(Vector2d bound){
        std::vector<BoundedPolynomial<degree,Type>> ps;
        for(auto& p:polys)
            p.bound(bound);
        collapse();
    }
};
template< uint degree, class Type>
CompoundBoundedPolynomial<degree,Type>
operator*(CompoundBoundedPolynomial<degree,Type> cbp, long double d){
    for(auto& p:cbp.polys)
        p.p.coeffs*=Type(d);
    return cbp;
}


template<unsigned int A, unsigned int B,class Type>
CompoundBoundedPolynomial<A+B,Type>
operator*(CompoundBoundedPolynomial<A,Type> a,
          CompoundBoundedPolynomial<B,Type> b){
    CompoundBoundedPolynomial<A+B,Type> out;
    for(auto ap:a.polys){
        for(auto bp:b.polys){
            out.add(ap*bp);
        }
    }
    return out;
}

template<unsigned int A, unsigned int B,class Type>
CompoundBoundedPolynomial<A+B,Type>
operator*(CompoundBoundedPolynomial<A,Type> a,
          Polynomial<B,Type> b)
{
    CompoundBoundedPolynomial<B,Type> c;
    c.add(BoundedPolynomial<B,Type>(a.span(),b));
    return a*c;
}

template<unsigned int A, unsigned int B,class Type>
CompoundBoundedPolynomial<(A<B)?B:A,Type>
operator+(CompoundBoundedPolynomial<A,Type> a,
          CompoundBoundedPolynomial<B,Type> b){
    CompoundBoundedPolynomial<(A<B)?B:A,Type> out;
    for(auto p:a.polys) out.add(p);
    for(auto p:b.polys) out.add(p);
    out.collapse();
    return out;
}
template<unsigned int A, unsigned int B,class Type>
CompoundBoundedPolynomial<(A<B)?B:A,Type>
operator-(CompoundBoundedPolynomial<A,Type> a,
          CompoundBoundedPolynomial<B,Type> b){
    CompoundBoundedPolynomial<(A<B)?B:A,Type> out;
    for(auto p:a.polys) out.add(p);
    for(auto p:b.polys) out.add(p);
    out.collapse();
    return out;
}




template<unsigned int S, class Type>
std::ostream& operator<<(std::ostream& os, CompoundBoundedPolynomial<S,Type> poly){

    return os<<poly.str();
}

} // end namespace cvl


