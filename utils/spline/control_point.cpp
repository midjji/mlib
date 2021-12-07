#include <mlib/utils/spline/control_point.h>
namespace cvl {

ControlPoint_::~ControlPoint_(){}

double* ControlPointWrapper::begin() const{return ptr->begin();}
std::string ControlPointWrapper::str() const{return ptr->str();}


ControlPointWrapper::ControlPointWrapper(){}
ControlPointWrapper::ControlPointWrapper(ControlPoint_* ptr):ptr(ptr){}
ControlPointWrapper::~ControlPointWrapper(){        delete ptr;    }




// Overloaded assignment
ControlPointWrapper::ControlPointWrapper(const ControlPointWrapper& obj)
{
    ControlPoint_* a=obj.ptr->clone();
    delete ptr;
    ptr=a;
}
ControlPointWrapper& ControlPointWrapper::operator= (ControlPoint_* obj)
{
    delete ptr;
    ptr=obj;
    return *this;
}
ControlPointWrapper& ControlPointWrapper::operator= (const ControlPointWrapper& obj)
{
    ControlPoint_* a=obj.ptr->clone();
    delete ptr;
    ptr=a;
    return *this;
}



}
