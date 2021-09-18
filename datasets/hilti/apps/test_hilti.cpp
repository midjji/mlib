#include <mlib/datasets/hilti/dataset.h>
#include <mlib/stream/buffer_node.h>
#include <mlib/opencv_util/imshow.h>
#include <iostream>
using std::cout;
using std::endl;

namespace cvl {





namespace hilti {




void show_hilti()
{
    cvl::hilti::Dataset ds;
    for(auto seq:ds.seqs){
        cout<<seq.samples()<<endl;
        for(int i=0;i<seq.samples();++i) {
            seq.sample(i)->show();
            wait(0);
        }
    }
}

}
}
















int main(){
    cvl::hilti::show_hilti();

    return 0;
}
