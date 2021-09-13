#include <iostream>
#include "editor.h"

#include <QMenuBar>
#include <QStatusBar>

#include <pset_widget.h>
#include <pset.h>

using std::cout;
using std::endl;
namespace cvl {


ParameterEditor::ParameterEditor(QWidget *parent)
    : QMainWindow(parent),
      central(new ParamSetWidget(this))
{

    setCentralWidget(central);


    //setCentralWidget(treeView);



    //QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    //fileMenu->addAction(newAct);
    //fileMenu->addAction(openAct);
    //fileMenu->addAction(saveAct);
    //statusBar()->showMessage(tr("Ready"));
}

void ParameterEditor::set(std::shared_ptr<ParamSet> p){
    central->set(p);
}


}
