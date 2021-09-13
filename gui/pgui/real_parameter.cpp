
#include <real_parameter.h>

namespace cvl {



RealParameter::RealParameter(
        double default_value,
        std::string name,
        std::string group,
        std::string desc,
        double minv,
        double maxv):
    Parameter(name, group, desc),
    minv(minv), maxv(maxv),
    value_(default_value), new_value(default_value)
{}

bool RealParameter::ranged() const{
    return (minv==std::numeric_limits<double>::min() &&
            maxv==std::numeric_limits<double>::max());
}

// the user value
double RealParameter::value() const{return value_;}

// the user selects when to update
bool RealParameter::update_value() {
    std::unique_lock<std::mutex> ul(mtx);
    bool c=changed();
    value_=double(new_value);
    current=true;
    return c;
}

// The value the gui wants to set
double RealParameter::gui_value()const{return new_value;}
void RealParameter::set_value(double value)
{
    std::unique_lock<std::mutex> ul(mtx);
    double nv=validate(value);
    if(nv==new_value) return;
    new_value=nv;
    current=false;
}
bool RealParameter::changed() const{return value_!=new_value;}
double RealParameter::validate(double a) const
{
    if(a<minv) return minv;
    if(a<maxv) return a;
    return maxv;
}

}


