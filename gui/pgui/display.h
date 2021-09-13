#pragma once
#include <QWidget>



namespace cvl {
class Label;
class ParamSet;

class ParamDisplayWidget: public QWidget{
    Q_OBJECT
public:
    ParamDisplayWidget(ParamSet* ps,
                       QWidget* parent = nullptr);
    Label* name;
    Label* desc;

};

}
