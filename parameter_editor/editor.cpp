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
      central(new PSetWidget(this))
{

    setCentralWidget(central);

    QMenu* fileMenu = menuBar()->addMenu(tr("&File"));
    //fileMenu->addAction([](){});
    //fileMenu->addAction(load);
    //statusBar()->showMessage(tr("Ready"));
}

void ParameterEditor::set(std::shared_ptr<PSet> p){
    central->set(p);
}
void ParameterEditor::save(std::string path){
    if(path==""); // open a file dialogue...
}
void ParameterEditor::load(std::string path){
    if(path==""); // open a file dialogue

}

}
