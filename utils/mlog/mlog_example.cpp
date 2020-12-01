#include "log.h"
#include <iostream>
#include <thread>
using std::cout;using std::endl;

using namespace cvl;
void function_a(){
    mlog()<<"msg in function a"<<endl;
}
void function_b(){
    mlog()<<"msg in function b"<<endl;
}
class AClass{
public:
    AClass(){}
    void fun(){
        mlog()<<"msg in AClass::fun"<<endl;
    }
};

class BClass{
public:
    BClass(){}
    void afun(){
        mlog()<<"msg in AClass::fun"<<endl<<"\n"<<"tstse "<<"\nmore...";
    }
};
void test(){
    mlog().set_thread_name("test thread");
    AClass b;
    b.fun();
}




int main(){






    while(true){
        if(true){

            mlog()<<endl;
            mlog()<<"mlog here"<<endl;



            mlog()<<"test 1"<<" and more test1"<< " and even more test 1"<<endl;


            function_a();

            function_b();
            AClass as;
            as.fun();

            std::thread thr(test);

            thr.join();

            printf(mlog(),"djaksdjflasdjflasdköfj %05d.txt\0\n",10);
        }

        break;
    }




    return 0;
}
