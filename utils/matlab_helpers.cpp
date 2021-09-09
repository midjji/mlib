#include <mlib/utils/matlab_helpers.h>
namespace mlib {
template<class T> std::string matlab_vector_impl(const std::vector<T>& vs, int precision){
    std::stringstream ss;
    ss<<std::setprecision(precision);
    ss<<"[";
    for(uint i=0;i<vs.size();++i){
        ss<<vs[i];
        if(i!=(vs.size()-1))
            ss<<"; ";
        if(i>40)
            if((i%50)==0)
            ss<<"\n";
    }
    ss<<"]";
    return ss.str();
}
std::string matlab_vector(const std::vector<float>& vs, int precision)
{
return matlab_vector_impl(vs,precision);
}
std::string matlab_vector(const std::vector<double>& vs, int precision)
{
return matlab_vector_impl(vs,precision);
}
std::string matlab_vector(const std::vector<long double>& vs, int precision)
{
return matlab_vector_impl(vs,precision);
}
}
