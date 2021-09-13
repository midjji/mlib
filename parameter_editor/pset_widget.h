#pragma once
#include <QWidget>
#include <memory>
class QTreeView;
class QGridLayout;

namespace cvl {


class ParamTree;
class ParamSet;

class ParamSetDisplayWidget;

class ParamSetWidget: public QWidget
{
    Q_OBJECT
public:
    using Layout=QGridLayout*;
    ParamSetWidget(QWidget* parent = nullptr);
    ParamTree* tree;
    ParamSetDisplayWidget* display;

    // once this thing gets the parametset,
    // the parameter set may no longer change?
    void set(std::shared_ptr<ParamSet> ps);

void set_display(std::shared_ptr<ParamSet> ps);

};

}
