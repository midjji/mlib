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

    virtual void loop() override
    {
        while(!ready){
            mlib::sleep_ms(10);
        }

        for(Output& out:iterable){
            mlib::ScopedDelay sd(1e6/framerate);
            this->push_output(out);
        }
        Node<Source<Output>,NoSink>::running=false;
    }
    using Input = typename Node<Source<Output>,NoSink>::Input;
    virtual bool process([[maybe_unused]] Input& input,
                         [[maybe_unused]] Output& output) override{return false;};
    std::atomic<bool> ready{false};

};

}
