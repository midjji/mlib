#pragma once
#include <QWidget>
#include <pset.h>
class Label;
namespace cvl {


class ParamDisplayWidget: public QWidget{
    Q_OBJECT
public:
    ParamDisplayWidget(ParamSet* ps,
                       QWidget* parent = nullptr);
    Label* name;
    Label* desc;

};

}
