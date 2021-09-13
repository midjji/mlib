#pragma once
#include <display.h>
QT_BEGIN_NAMESPACE
class QTreeView; //forward declarations
//class QStandardItem;// add latter...
QT_END_NAMESPACE
namespace cvl {


class ParamTree;
class ParamSet;


class ParamWidget: public QWidget
{
    Q_OBJECT
public:
    ParamWidget(QWidget* parent = nullptr);
    ParamTree* tree;
    ParamDisplayWidget* display;
    // once this thing gets the parametset,
    // the parameter set may no longer change?
    void set(ParamSet* ps);

};

}
