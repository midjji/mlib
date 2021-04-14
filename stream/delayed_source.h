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
        Iterable(iterable){}

    virtual void loop() override  {
        for(Output& out:iterable){
            //mlib::ScopedDelay sd(1e6/framerate);
            this->push_output(out);
        }
    }
};

}
