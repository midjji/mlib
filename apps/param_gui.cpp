#include <QApplication>
#include <mlib/utils/mlog/log.h>

struct ParamSet{};
struct Parameter{};
// not copy constructible, not singleton,
struct Parametrized{};

struct A: public Parametrized{
    double a;
    double ab; // double in range
    int ai;
    int abi;// int in range, also covers enum

    Parameter parameters;
};

void treegui



int main(){
    return 0;
}
