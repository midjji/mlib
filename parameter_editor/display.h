#pragma once
#include <QWidget>
#include <memory>
#include <QScrollArea>

namespace cvl {
class Label;
class PSet;

class PSetDisplayWidget: public QWidget{
    Q_OBJECT
public:
    // Only shows the top level,
    // so the name and desc, plus the params in groupss
    PSetDisplayWidget(std::shared_ptr<PSet> ps,
                       QWidget* parent = nullptr);

private:
    std::shared_ptr<PSet> ps;

};



}
