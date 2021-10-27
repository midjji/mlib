
#include <enum_parameter.h>

namespace cvl {

EnumParameter::EnumParameter(
        int default_value,
        std::map<int, std::string> options,
        std::string name,
        std::string group,
        std::string desc
       ):
    Parameter(name, group, desc),
    value_(default_value), new_value(default_value), options(options)
{}

// the user value
int EnumParameter::value() const{return value_;}

// the user selects when to update
bool EnumParameter::update_value() {

    bool c=changed();
    value_=int(new_value);
    current=true;
    return c;
}
Parameter::type_t EnumParameter::type() const {return type_t::integer;}

// The value the gui wants to set
int EnumParameter::gui_value()const{return new_value;}
void EnumParameter::set_value(int value)
{

    if(value==new_value) return;
    new_value=value;
    current=false;
}
bool EnumParameter::changed() const{return value_!=new_value;}


}

