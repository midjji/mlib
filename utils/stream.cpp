#if 0
#include <mlib/utils/stream.h>

using namespace std;

StreamData::~StreamData(){}

void Stream::addOperator(std::shared_ptr<StreamOperator> so){
    if(!done)
        streamops.push_back(so);
    else
        std::cout<<"Added "+so->name()+" operation to active stream!"<<std::endl;
}
StreamOperator::~StreamOperator(){}
void Stream::run(){
    while(!done){
        std::shared_ptr<StreamData> sd=nullptr;
        for(auto op:streamops){
            cout<<"op: "<<op->name()<<" begin"<<endl;

            // multiplex? later
            auto state=(*op)(sd);
            if(state==STATE::BREAK){

                done=true;
                break;
            }
            if(state==STATE::CONTINUE){
                cout<<"op: "<<op->name()<<" Continue"<<endl;
                break;
            }
            if(state==STATE::NORMAL){// nop
                cout<<"op: "<<op->name()<<" NORMAL"<<endl;
                continue;
            }
            cout<<"op: "<<op->name()<<" BREAK"<<endl;

        }
    }
    for(auto op:streamops){op->stop();}
}
#endif
