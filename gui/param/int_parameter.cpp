
#include <int_parameter.h>

namespace cvl {



IntParameter::IntParameter(
        int default_value,
        std::string name,
        std::string group,
        std::string desc,
        int minv,
        int maxv):
    Parameter(name, group, desc),
    minv(minv), maxv(maxv),
    value_(default_value), new_value(default_value)
{}

bool IntParameter::ranged() const{
    return (minv==std::numeric_limits<int>::min() &&
            maxv==std::numeric_limits<int>::max());
}

// the user value
int IntParameter::value() const{return value_;}

// the user selects when to update
bool IntParameter::update_value() {
    std::unique_lock<std::mutex> ul(mtx);
    bool c=changed();
    value_=int(new_value);
    current=true;
    return c;
}
Parameter::type_t IntParameter::type() const {return type_t::integer;}

// The value the gui wants to set
int IntParameter::gui_value()const{return new_value;}
void IntParameter::set_value(int value)
{
    std::unique_lock<std::mutex> ul(mtx);
    int nv=validate(value);
    if(nv==new_value) return;
    new_value=nv;
    current=false;
}
bool IntParameter::changed() const{return value_!=new_value;}
int IntParameter::validate(int a) const
{
    if(a<minv) return minv;
    if(a<maxv) return a;
    return maxv;
}

}

