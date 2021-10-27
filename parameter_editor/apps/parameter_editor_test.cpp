#include <QApplication>
#include <mlib/parameter_editor/editor.h>
#include <mlib/param/pset.h>
#include <mlib/param/int_parameter.h>
using namespace cvl;

std::shared_ptr<PSet> test_PSet(){
    auto top=std::make_shared<PSet>("top","top desc");
    top->param<IntParameter>("top",1,"top 1");
    top->param<IntParameter>("top 2", 1,"top 2");
    top->add("subseta", std::make_shared<PSet>("a",""));
    auto b=std::make_shared<PSet>("b");
    b->param<IntParameter>("top4",2,"top 4","group 1");
    b->param<IntParameter>("top 5", 2,"top 5","group 1");
    b->param<IntParameter>("top 6", 2,"top 6","group 2");
    top->add("subsetb", b);
    b->add("subsetc", std::make_shared<PSet>("c"));
    b->add("subsetd",std::make_shared<PSet>("d","ddesc"));
    return top;
}

std::shared_ptr<PSet> load_pset(std::string path="test.parameterset")
{
    std::shared_ptr<PSet> ps=std::make_shared<PSet>();
   // ps->load(path);
    return ps;
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ParameterEditor w;
    w.set(test_PSet());
    w.show();
    return a.exec();
}

