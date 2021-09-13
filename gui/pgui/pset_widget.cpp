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

#include <pset_widget.h>
#include <pset.h>
#include <display.h>
#include <param_tree.h>



using std::cout;
using std::endl;
namespace cvl {


ParamSetWidget::ParamSetWidget(QWidget* parent):
    QWidget(parent),
    tree(new ParamTree(this)),
    display(new ParamSetDisplayWidget(nullptr, this))
{
    // no need for this to be a member of ParamSetWidget too,
    // get it from widget instead if needed...
    // auto does setlayout when called with parent this
    auto layout=new QGridLayout(this);
    layout->addWidget(tree,0,0);
    layout->addWidget(display,0,1);

}
void ParamSetWidget::set(std::shared_ptr<ParamSet> ps) {
    tree->clear();
    tree->add(ps);
}
}
