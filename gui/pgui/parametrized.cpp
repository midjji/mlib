#include <parametrized.h>
#include <pset.h>
namespace cvl {
Parametrized::Parametrized(std::string name,std::string desc):
    param(std::make_shared<ParamSet>(name,desc)){}
Parametrized::~Parametrized(){        param->alive=false;    }
std::shared_ptr<ParamSet> Parametrized::params(){return param;}
void Parametrized::update_all(){
    param->update_all();
}
IntParameter* Parametrized::pint(
        int default_value,
        std::string name,
        std::string group,
        std::string desc,
        int minv,
        int maxv) {
    auto p=new IntParameter(default_value,name, group,desc,minv,maxv);
            param->add_parameter(p);
    return p;
}
RealParameter* Parametrized::preal(double default_value,
                                   std::string name,
                                   std::string group,
                                   std::string desc,
                                   double minv,
                                   double maxv) {
    auto p=new RealParameter(default_value,name, group,desc,minv,maxv)
            param->add_Parameter(p);
    return p;
}
}
