#pragma once
#include <string>
#include <mlib/utils/cvl/matrix.h>

// Todo, replace with std::any?
namespace cvl {
struct ControlPoint_{
    virtual ~ControlPoint_();
    virtual double* begin()=0;
    virtual const double* begin() const=0;
    virtual std::string str() const=0;
    virtual ControlPoint_* clone() const=0; // deep copy
};

struct ControlPointWrapper
{
    ControlPoint_* ptr=nullptr;

    ControlPointWrapper();
    ControlPointWrapper(ControlPoint_* ptr);
    ~ControlPointWrapper();
    virtual double* begin() const;
    virtual std::string str() const;

    // Overloaded assignment
    ControlPointWrapper& operator= (ControlPoint_* obj);
    ControlPointWrapper& operator= (const ControlPointWrapper& obj);
    ControlPointWrapper(const ControlPointWrapper& obj);
};
template<int Dimensions>
struct ControlPoint: public ControlPoint_
{
    ControlPoint(Vector<double,Dimensions> x):x(x){}
    Vector<double, Dimensions> x;
    double* begin() override{return x.begin();}
    const double* begin() const override{return x.begin();}

    std::string str() const override{
        std::stringstream ss;
        ss<<x;
        return ss.str();
    }
    ControlPoint_* clone() const override{
        return new ControlPoint<Dimensions>(x);
    }
};


}
