#include <iostream>

#include <QVBoxLayout>

#include <param_widget.h>
#include <int_parameter.h>
#include <real_parameter.h>
#include <parameter.h>

#include <label.h>



namespace cvl
{
QWidget* display(IntParameter* p, QWidget* parent)
{

}
QWidget* display(RealParameter* p, QWidget* parent)
{

}
QWidget* display(Parameter* p, QWidget* parent)
{

    using intp=IntParameter*;
    using realp=RealParameter*;

    switch (p->type())
    {
    case Parameter::type_t::integer: return display(intp(p),parent);
    case Parameter::type_t::real: return display(realp(p),parent);
    default:
        std::cout<<"Parameter type not supported, returning nullptr"<<std::endl;
        return nullptr;
    }
}
class GroupWidget:public QWidget{
public:
    Label name;
    GroupWidget(std::vector<Parameter*> group,
                std::string groupname,
                QWidget* parent):
        name(groupname,parent){
        auto layout=new QVBoxLayout(this); // also sets the layout

        for(auto* p:group){
            layout->addWidget(display(p,this));
        }
    } // false positive for leak
};

QWidget* display_group(std::vector<Parameter*> group, std::string groupname, QWidget* parent)
{
    return new GroupWidget(group,groupname,parent);
}
}
