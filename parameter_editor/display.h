#pragma once
#include <QWidget>
#include <memory>
#include <QScrollArea>

namespace cvl {
class Label;
class ParamSet;

class ParamSetDisplayWidget: public QWidget{
    Q_OBJECT
public:
    // Only shows the top level,
    // so the name and desc, plus the params in groupss
    ParamSetDisplayWidget(std::shared_ptr<ParamSet> ps,
                       QWidget* parent = nullptr);


};



}
