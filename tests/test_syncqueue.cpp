#include <mlib/utils/cvl/syncque.h>
int main(){
    cvl::SyncQue<double> sq;
    sq.push(1);
    double a;
    sq.try_pop(a);
    assert(a==1);
    sq.stop();
    return 0;
}
