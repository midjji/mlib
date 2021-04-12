#include <vector>
#include <iostream>
#include <sstream>
#include <mlib/utils/symbolic.h>

namespace cvl {

Sym::Sym(std::string n, int exponent){
    comps[n]=exponent;
}
std::string Sym::str() const{
    std::stringstream ss;
    for(const auto& a:comps){
        if(a.first=="I") continue;

        ss<<a.first;
        if(a.second>1)
            ss<<"^"<<a.second;

    }
    return ss.str();
}
std::string Sym::hash() const{
    std::stringstream ss;
    for(const auto& a:comps){
        ss<<a.first;
        ss<<"^"<<a.second;
    }
    return ss.str();
}
void Sym::simplify(){
    for(auto it=comps.begin();it!=comps.end();++it)
        if(it->second==0)
            comps.erase(it);
}
Sym& Sym::operator*=(Sym a){
    for(const auto& v:a.comps)
        comps[v.first]+=v.second; // get zero initialized
    return *this;
}
Sym& Sym::operator/=(Sym a){
    for(const auto& v:a.comps)
        comps[v.first]-=v.second; // get zero initialized
    return *this;
}

Sym operator*(Sym a, Sym b){
    Sym c=a;
    c*=b;
    return c;
}
Sym operator/(Sym a, Sym b){
    Sym c=a;
    c/=b;
    return c;
}
bool operator<(Sym a, Sym b){
    return a.hash()<b.hash();
}

Symb::Symb(double d){
    if(d!=0.0)
        koeffs[Sym("I")]=d;
}

Symb::Symb(Sym a, double k){
    koeffs[a]=k;
}

void Symb::clear_zeros(){

    std::map<Sym,double> ks;
    for(const auto& k:koeffs)
        if(k.second!=0.0)
            ks[k.first]=k.second;
    koeffs=ks;


}
bool operator<([[maybe_unused]] Symb a, [[maybe_unused]] Symb b){
    return false;
}
Symb operator-(Symb s){
    Symb r=s;
    for(auto& [toss, k]:r.koeffs)
        k=-1;
    return r;
}

std::string Symb::str(){


    clear_zeros();
    std::stringstream ss;
    bool first=true;
    bool plus=false;
    for(const auto& [sym,k]:koeffs){
        double v=k;
        if(!first){plus=true;
            if(k>0)
                ss<<" + ";
            else{
                ss<<" - ";
                v=-v;
            }

        }
        first=false;
        if(std::abs(k-1.0)>1e-12)
            ss<<v;
        ss<<sym.str();
    }
    if(ss.str().size()==0)
        ss<<"0";


    if(plus){
        std::stringstream ss2;
        ss2<<"("<<ss.str()<<")";
        return ss2.str();
    }
    return ss.str();
}
Symb& Symb::operator+=(Symb b){
    Symb c= *this + b;
    koeffs=c.koeffs;
    return *this;
}
Symb& Symb::operator*=(Symb b){
    Symb c= *this * b;
    koeffs=c.koeffs;
    return *this;
}

Symb operator+(Symb a, Symb b){

    Symb c=a;
    for(const auto& v:b.koeffs){
        c.koeffs[v.first]+=v.second; // get zero initialized
    }
    c.clear_zeros();
    return c;
}
Symb operator*(Symb as, Symb bs)
{

    as.clear_zeros();
    bs.clear_zeros();
    std::vector<Sym> ss;
    std::vector<double> ks;
    for(const auto& a:as.koeffs)
        for(const auto& b:bs.koeffs){
            ss.push_back(a.first*b.first);
            ks.push_back(a.second*b.second);
        }

    Symb cs;
    for(uint i=0;i<ss.size();++i){
        cs.koeffs[ss[i]]+=ks[i];
    }

    return cs;
}
Symb operator*(Symb as, double d){
    for(auto& [a,k]:as.koeffs)
        k*=d;
    return as;
}

bool operator==(Symb s, double d){
    if(d!=0.0) return false;
    s.clear_zeros();
    return s.koeffs.size()==0;
}

std::ostream& operator<<(std::ostream& os, cvl::Sym s){
    return os<<s.str();
}
std::ostream& operator<<(std::ostream& os, cvl::Symb s){
    return os<<s.str();
}


} // end namespace cvl






