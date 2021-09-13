#include <QApplication>
#include <editor.h>
#include <pset.h>
using namespace cvl;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ParameterEditor w;

    auto top=ParamSet::create("top","top desc");
    top->add(ParamSet::create("a",""));
    auto b=ParamSet::create("b");
    top->add(b);
    b->add(ParamSet::create("c"));
    b->add(ParamSet::create("d","ddesc"));
    w.set(top);
    w.show();
    return a.exec();
}

