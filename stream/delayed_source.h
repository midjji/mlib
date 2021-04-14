#pragma once
#include <mlib/stream/node.h>
#include <mlib/utils/mlibtime.h>

namespace cvl {
template<class Output, class Iterable>
class DelayedSource: public Node<Source<Output>,NoSink>{
    double framerate;
    Iterable iterable;
public:

    DelayedSource(double framerate,
                  Iterable& iterable):
        framerate(framerate),
        iterable(iterable){}

    virtual void loop() override  {
        for(Output& out:iterable){
            mlib::ScopedDelay sd(1e9/framerate);
            this->push_output(out);
            //mlib::sleep_us(1e6/framerate);
        }
        std::cout << "done with iterable" << std::endl;
    }        
};

}
