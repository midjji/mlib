#include <mlib/utils/serialization.h>
#include <assert.h>
#include <iostream>
using std::cout;using std::endl;
using namespace mlib;
int main(){
    std::string indata="bla bla bla, \nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";

    bool good_write=verified_write(indata,"test.txt");
    assert(good_write);
    if(!good_write)
        cout<<"write failed"<<endl;
    std::string tmp;
    bool good_read=verified_read(tmp,"test.txt");
    if(!good_read)
        cout<<"read failed"<<endl;
    assert(good_read);
    assert(tmp==indata);
    if(tmp!=indata)
        cout<<"read failed"<<endl;
    return 0;
}
