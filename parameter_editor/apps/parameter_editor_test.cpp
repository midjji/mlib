#include <QApplication>
#include <mlib/parameter_editor/editor.h>
#include <mlib/param/pset.h>
#include <mlib/param/int_parameter.h>
using namespace cvl;

std::shared_ptr<ParamSet> test_paramset(){
    auto top=ParamSet::create("top","top desc");
    top->add(new IntParameter(1,"top 1"));
    top->add(new IntParameter(1,"top 2"));
    top->add(ParamSet::create("a",""));
    auto b=ParamSet::create("b");
    b->add(new IntParameter(2,"top 4","group 1"));
    b->add(new IntParameter(2,"top 5","group 1"));
    b->add(new IntParameter(2,"top 6","group 2"));
    top->add(b);
    b->add(ParamSet::create("c"));
    b->add(ParamSet::create("d","ddesc"));
    return top;
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ParameterEditor w;
    w.set(test_paramset());
    w.show();
    return a.exec();
}

