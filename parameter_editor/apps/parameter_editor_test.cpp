#include <QApplication>
#include <mlib/parameter_editor/editor.h>
#include <mlib/param/pset.h>
#include <mlib/param/int_parameter.h>
using namespace cvl;

std::shared_ptr<PSet> test_PSet(){
    auto top=PSet::create("top","top desc");
    top->add(new IntParameter(1,"top 1"));
    top->add(new IntParameter(1,"top 2"));
    top->add(PSet::create("a",""));
    auto b=PSet::create("b");
    b->add(new IntParameter(2,"top 4","group 1"));
    b->add(new IntParameter(2,"top 5","group 1"));
    b->add(new IntParameter(2,"top 6","group 2"));
    top->add(b);
    b->add(PSet::create("c"));
    b->add(PSet::create("d","ddesc"));
    return top;
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ParameterEditor w;
    w.set(test_PSet());
    w.show();
    return a.exec();
}

