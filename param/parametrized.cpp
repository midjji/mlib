#include <parametrized.h>
#include <pset.h>
namespace cvl {
Parametrized::Parametrized(std::string name,std::string desc):
    param(std::make_shared<PSet>(name,desc)){}
Parametrized::~Parametrized(){           }
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
   return (IntParameter*)(param->param<IntParameter>(name, default_value, name, group, desc, minv, maxv).get());

}
RealParameter* Parametrized::preal(double default_value,
                                   std::string name,
                                   std::string group,
                                   std::string desc,
                                   double minv,
                                   double maxv) {
   return  (RealParameter*)(param->param<RealParameter>(name, default_value,name, group,desc,minv,maxv).get());

}
void Parametrized::add(std::string unique_identifier,
                       std::shared_ptr<PSet> p){
    param->add(unique_identifier, p);
}
}
