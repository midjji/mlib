#include <opencv2/highgui.hpp>
#include <mlib/datasets/hilti/dataset.h>

#include <mlib/opencv_util/imshow.h>


namespace cvl {


namespace hilti {


void inspect(){
    Dataset ds;

    auto seq=ds.seqs[0]; // construction site, easy




    for(int i=0;i<seq->samples();i+=1){
        auto sample=seq->sample(i);
        for(int i=0;i<5;++i)
        {
            if(!sample->has(i)) continue;
            imshow(sample->rgb(i), sample->num2name(i));
        }
        if(sample->has(5)) imshow(sample->rgb(5));
        cv::waitKey(0);
    }


}
}
}


int main()
{
    cvl::hilti::inspect();
}
