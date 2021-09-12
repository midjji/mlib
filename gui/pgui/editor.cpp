#include <iostream>
#include "editor.h"

#include <QMenuBar>
#include <QStatusBar>

#include <param_widget.h>

using std::cout;
using std::endl;

ParameterEditor::ParameterEditor(QWidget *parent)
    : QMainWindow(parent),
      central(new ParamWidget(this))
{
    auto central=new ParamWidget(this);
    ParamSet* top=new ParamSet("top");
    ParamSet* psa=new ParamSet("a");
    ParamSet* psb=new ParamSet("b");
    top->pss.push_back(psa);
    top->pss.push_back(psb);
    central->set(top);
    setCentralWidget(central);


    //setCentralWidget(treeView);



    //QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    //fileMenu->addAction(newAct);
    //fileMenu->addAction(openAct);
    //fileMenu->addAction(saveAct);
    //statusBar()->showMessage(tr("Ready"));

}


