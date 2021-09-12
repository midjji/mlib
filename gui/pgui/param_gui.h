#pragma once
#include <params.h>
#include <QWidget>
#include <QStandardItem>

QT_BEGIN_NAMESPACE
class QTreeView; //forward declarations
class QStandardItemModel;
class QItemSelection;
class QLayout;
class QLabel;
//class QStandardItem;// add latter...
QT_END_NAMESPACE



class ParamItem:public QStandardItem{
public:
    ParamItem(ParamSet* ps);
    std::string name="pi";
    ParamSet* ps;
};

class ParamTree:public QWidget
{
    Q_OBJECT
public:
    QTreeView* tree;
    QStandardItemModel* item_model();
    ParamItem* selected_item();
    ParamTree(QWidget* parent = nullptr);
    void clear();
    void add(ParamSet* ps);
private slots:

    void selected(
            //const QItemSelection &newSelection,
            //const QItemSelection &oldSelection
            );
};

