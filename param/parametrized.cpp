#include <parametrized.h>
#include <pset.h>
namespace cvl {
Parametrized::Parametrized(std::string name,std::string desc):
    param(std::make_shared<PSet>(name,desc)){}
Parametrized::~Parametrized(){        param->set_alive(false);    }
std::shared_ptr<PSet> Parametrized::params(){return param;}
void Parametrized::update_all(){
    param->update_all();
}
IntParameter* Parametrized::pint(
        int default_value,
        std::string name,
        std::string group,
        std::string desc,
        int minv,
        int maxv)
{
   return param->add<IntParameter>(name, default_value, name, group, desc, minv, maxv);

}
RealParameter* Parametrized::preal(double default_value,
                                   std::string name,
                                   std::string group,
                                   std::string desc,
                                   double minv,
                                   double maxv) {
   return  param->add<RealParameter>(name, default_value,name, group,desc,minv,maxv);

}
void Parametrized::add(std::shared_ptr<PSet> p){
    param->add(p);
}
}
