#include <mlib/utils/kde.h>
#include <mlib/utils/mlog/log.h>
#include <algorithm>
#include <cmath>

namespace cvl{
KDE::KDE(const std::vector<double>& datas_):datas(datas_){
    // significantly improves numerics...
    std::sort(datas.begin(), datas.end(),[](double a, double b){return std::abs(a)<std::abs(b);});
    datas_sorted=datas;
    std::sort(datas_sorted.begin(), datas_sorted.end());
}
double KDE::operator()(double t, double sigma) const{
    double nf=(1.0/(sigma*std::sqrt(2*3.14159265359)))/double(datas.size());
    double output=0;
     double s2=-1.0/(2.0*sigma*sigma);
    for(double d:datas){
        double v=std::exp(s2*(t - d)*(t - d));
        output+=v;
    }
    return output*nf;
}
double KDE::low_bound(double ratio) const{
    if(datas.size()==0) return 0;    
    return datas_sorted.at(datas_sorted.size()*ratio);
}
double KDE::high_bound(double ratio) const{
    if(datas.size()==0) return 0;    
    return datas_sorted.at(datas_sorted.size()*ratio);
}

std::tuple<std::vector<double>, std::vector<double>>
KDE::plot(double low, double high, int N, double sigma) const
{
    std::vector<double> xs,ys;
    xs.reserve(N);
    ys.reserve(N);

    double delta=(high - low)/double(N);
    for(int i=0;i<N;++i){
        double t=i*delta + low;
        xs.push_back(t);
        ys.push_back(this->operator()(t,sigma));
    }
    return std::make_tuple(xs,ys);
}
std::tuple<std::vector<double>, std::vector<double>>
KDE::plot(int N, double sigma) const
{
    std::vector<double> xs,ys;
    xs.reserve(N);
    ys.reserve(N);
    double low=low_bound(0.01);
    double high=high_bound(0.99);

    double delta=(high - low)/double(N);
    for(int i=0;i<N;++i){
        double t=i*delta + low;
        xs.push_back(t);
        ys.push_back(this->operator()(t,sigma));
    }
    return std::make_tuple(xs,ys);
}
}

