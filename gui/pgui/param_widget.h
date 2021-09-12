#pragma once
#include <QMainWindow>
#include <QStandardItem>

#include <display.h>
QT_BEGIN_NAMESPACE
class QTreeView; //forward declarations
class QStandardItemModel;
class QItemSelection;
class QLayout;
class QLabel;
//class QStandardItem;// add latter...
QT_END_NAMESPACE
class ParamTree;

class ParamSet;


class ParamWidget: public QWidget
{
    Q_OBJECT
public:
    ParamWidget(QWidget* parent = nullptr);
    ParamTree* tree;
    ParamDisplayWidget display;
    // once this thing gets the parametset,
    // the parameter set may no longer change?
    void set(ParamSet* ps);

};

