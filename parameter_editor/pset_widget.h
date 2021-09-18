#pragma once
#include <QWidget>
#include <memory>
class QTreeView;
class QGridLayout;

namespace cvl {


class ParamTree;
class PSet;

class PSetDisplayWidget;

class PSetWidget: public QWidget
{
    Q_OBJECT
public:
    using Layout=QGridLayout*;
    PSetWidget(QWidget* parent = nullptr);
    ParamTree* tree;
    PSetDisplayWidget* display;

    // once this thing gets the parametset,
    // the parameter set may no longer change?
    void set(std::shared_ptr<PSet> ps);

void set_display(std::shared_ptr<PSet> ps);

};

}
