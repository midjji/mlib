#include <iostream>

#include <QTreeView>
#include <QStandardItemModel>
#include <QItemSelectionModel>
#include <QTreeView>
#include <QStandardItemModel>
#include <QItemSelectionModel>
#include <QGridLayout>
#include <QLabel>
#include <QMenuBar>
#include <QStatusBar>
#include <QHeaderView>
#include <QScrollArea>

#include <pset_widget.h>
#include <pset.h>
#include <display.h>
#include <param_tree.h>
#include <param_tree_item.h>



using std::cout;
using std::endl;
namespace cvl {


PSetWidget::PSetWidget(QWidget* parent):
    QWidget(parent),
    tree(new ParamTree(this)),
    display(new PSetDisplayWidget(nullptr, this))
{

    // no need for this to be a member of PSetWidget too,
    // get it from widget instead if needed...
    // auto does setlayout when called with parent this
    auto* layout=new QGridLayout(this);

    layout->addWidget(tree,0,0);
    layout->addWidget(display,0,1);
    layout->setColumnStretch(0,0);
    layout->setColumnStretch(1,0);
    layout->setRowStretch(0,0);
    layout->setRowStretch(2,1);
    layout->setColumnStretch(2,1);

    layout->setSizeConstraint(QLayout::SetMinimumSize);





    connect(tree->tree->selectionModel(),
            &QItemSelectionModel::selectionChanged,
            this,
            [&](){
        set_display(tree->selected_item()->ps);
    });

}
void PSetWidget::set_display(std::shared_ptr<PSet> ps){
    layout()->removeWidget(display);
    delete display;
    display=new PSetDisplayWidget(ps, this);

    Layout(layout())->addWidget(display,0,1);

}


void PSetWidget::set(std::shared_ptr<PSet> ps) {
    tree->clear();
    tree->add(ps);
    set_display(ps);

}
}
