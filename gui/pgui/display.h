#pragma once
#include <QWidget>
#include <params.h>
class Label;
class ParamDisplayWidget: public QWidget{
    Q_OBJECT
public:
    ParamDisplayWidget(ParamSet* ps,
                       QWidget* parent = nullptr);
    Label* name;
    Label* desc;

};

