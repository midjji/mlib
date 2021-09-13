#pragma once
#include <QWidget>
#include <memory>
class QTreeView;

namespace cvl {


class ParamTree;
class ParamSet;
class ParamDisplayWidget;

class ParamWidget: public QWidget
{
    Q_OBJECT
public:
    ParamWidget(QWidget* parent = nullptr);
    ParamTree* tree;
    ParamDisplayWidget* display;
    // once this thing gets the parametset,
    // the parameter set may no longer change?
    void set(std::shared_ptr<ParamSet> ps);

};

}
