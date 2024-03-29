#include <iostream>

#include <QVBoxLayout>

#include <param_widget.h>
#include <int_parameter.h>
#include <real_parameter.h>
#include <parameter.h>

#include <mlib/gui/label.h>
#include <QFormLayout>
#include <QLineEdit>
#include <QIntValidator>
#include <QDoubleValidator>

#include <sstream>
#include <mlib/utils/mlog/log.h>

namespace cvl
{

namespace  {
QString qstr(int i){    std::stringstream ss;    ss<<i;    return ss.str().c_str();}
QString qstr(double d){    std::stringstream ss;    ss<<d;    return ss.str().c_str();}
}

class IntParameterWidget:public QWidget{
public:
    QLineEdit* edit;
    std::shared_ptr<IntParameter> p;
    IntParameterWidget(std::shared_ptr<IntParameter> p,
                       QWidget* parent):
        QWidget(parent), p(p)
    {
require(p!=nullptr,"");
        auto* layout =new QHBoxLayout(this);
        layout->setMargin( 0 );
        layout->setSpacing( 0 );

        layout->addWidget(new Label(p->name,this));
        edit=new QLineEdit(this);
        edit->setAlignment(Qt::AlignRight);
        edit->setText(qstr(p->gui_value()));
        edit->setValidator(new QIntValidator(p->minv, p->maxv, edit));
        layout->addWidget(edit);
        if(p->desc.size()>0)
            setToolTip(p->desc.c_str());

        connect(edit,
                &QLineEdit::editingFinished,
                this,
                [&,p]{ /*or this->p... the reference stops existing without copy!*/p->set_value(edit->text().toInt()); });


    }
};

class RealParameterWidget:public QWidget{
public:
    QLineEdit* edit;
    std::shared_ptr<RealParameter> p;
    RealParameterWidget(std::shared_ptr<RealParameter> p,
                        QWidget* parent):
        QWidget(parent), p(p)
    {

        require(p!=nullptr,"");
        auto* layout =new QHBoxLayout(this);
        layout->setMargin( 0 );
        layout->setSpacing( 0 );

        layout->addWidget(new Label(p->name,this),0);
        edit=new QLineEdit(this);
        edit->setAlignment(Qt::AlignRight);
        edit->setText(qstr(p->gui_value()));
        edit->setValidator(new QDoubleValidator(p->minv, p->maxv,6, edit));
        layout->addWidget(edit,0);
        if(p->desc.size()>0)
            setToolTip(p->desc.c_str());

        connect(edit,
                &QLineEdit::editingFinished,
                this,
                [&,p]{ p->set_value(edit->text().toDouble()); });

    }
};


QWidget* display(std::shared_ptr<Parameter> p, QWidget* parent)
{

    if(p->is_int()) return new IntParameterWidget(std::dynamic_pointer_cast<IntParameter>(p),parent);
    if(p->is_double()) return new RealParameterWidget(std::dynamic_pointer_cast<RealParameter>(p),parent);
    wtf();
    return nullptr;

}
class GroupWidget:public QFrame{
public:

    GroupWidget(std::vector<std::shared_ptr<Parameter>> group,
                std::string groupname,
                QWidget* parent):QFrame(parent)
    {
        auto* layout=new QVBoxLayout(this); // also sets the layout

        setFrameStyle(QFrame::Panel | QFrame::Raised);



        layout->setSizeConstraint(QLayout::SetMinimumSize);
        layout->addWidget(new Label(groupname, this),0,Qt::AlignmentFlag::AlignLeft|Qt::AlignmentFlag::AlignTop);

        for(const auto& p:group){
            layout->addWidget(display(p,this),0,Qt::AlignmentFlag::AlignLeft|Qt::AlignmentFlag::AlignTop);
        }
    } // false positive for leak
};

QWidget* display_group(std::vector<std::shared_ptr<Parameter>> group, std::string groupname, QWidget* parent)
{
    return new GroupWidget(group,groupname,parent);
}
}
