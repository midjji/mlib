#pragma once
#include <parametrized.h>


namespace cvl {


struct A: public Parametrized
{
// Parametrized owns paramset.

    IntParameter* a= new_int(100,"num features", "group0","how many features");
    IntParameter* b= new_int(100,"num features", "group0","how many features");
    A(){

    }




    void operator()(){
        a->update_value();
        for(int i=0;i<features->value();++i);

    }
};


}
