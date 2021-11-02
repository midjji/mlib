#include <algorithm>
#include "klt/cornerdetector.h"

#include "klt/internal/multiply.h"
#include "klt/internal/non_maxima_supression.h"
#include "klt/internal/feature_mask_image.h"
#include "klt/internal/harris.h"
#include <mlib/utils/mlibtime.h>



namespace klt
{





namespace
{
inline void cap(int& a, int l, int h){

    if(a<l) a=l;
    if(a>h) a=h;
};
}

std::vector<cvl::Vector2f>
HarrisDetector::candidates(
        const Texture< float, false >& corner_scores_host,
        float min_quality)
{

    mhere();
    auto& timer=ntp["inner"];
    timer.tic();


    std::vector<HarrisDetector::Candidate > cs;
    cs.reserve(1e4);

    if(min_quality<0) min_quality=1e-6;

    int rborder=20;
    if(corner_scores_host.rows()*0.5>20) rborder=corner_scores_host.rows()*0.05;
    int cborder=20;
    if(corner_scores_host.cols()*0.5>20) cborder=corner_scores_host.cols()*0.05;

    float out=0;
    for ( int row=rborder; row<corner_scores_host.rows()-rborder; ++row )
        for ( int col=cborder; col<corner_scores_host.cols()-cborder; ++col )
        {
            // std::cout<<row<<", "<<col<<" "<<std::endl;
            float score=corner_scores_host(row,col);
            if ( score <= min_quality ) continue;
            cs.emplace_back(row,col,score);
        }


    std::cout<<"candidates found: "<<cs.size()<<" "<<corner_scores_host.elements()<<out<<std::endl;
    // sort or randomize? either works, and higher score isnt noticeably better...
    // random then sort? anms?
    // should be anms with a max feature count score, but leave that for latter...

    std::sort( cs.begin(), cs.end() );


    std::vector<cvl::Vector2f> rets;rets.reserve(cs.size());
    for(const auto& c:cs){
        rets.emplace_back(c.row,c.col);
    }

    timer.toc();

    return rets;
}


HarrisDetector::HarrisDetector(std::string name) :
    cvl::Parametrized("Corner Detector", "Harris Corner detector, i.e. minimum structure tensor eigenvalue")
{
    min_score_=preal(1000,"Minimum eigenvalue", "",">0 is sensible", 0);
   // mlog()<<"min_score_: "<<min_score_->value()<<"\n";
    radius_=pint(10,"Nonmaxima supression radius", "","This is a prefilter, which should be applied before anms, sensible values are 3-20, but higher radii can be slow, ", 0,100);
   // mlog()<<"radius: "<<radius_->value()<<"\n";
}


void HarrisDetector::create_buffers(int rows, int cols)
{
    std::cout<<"creating bufferts"<<std::endl;
    corner_scores.resize_rc( rows,cols);
    corner_scores_extra.resize_rc( rows,cols);
    corner_scores_host.resize_rc(rows, cols);
    masked_corner_score.resize_rc(rows,cols);
    mask_host.resize_rc(rows, cols);
    mask_device.resize_rc( rows, cols);
    locked_keypoints_host.resize_array(20000);
    locked_keypoints_device.resize_array(20000);
    mhere();
}





void HarrisDetector::compute_mask(const std::vector<cvl::Vector2f>& kps) {


    set2(mask_device,1);
    mhere();
    if(kps.size()==0) return;


    locked_keypoints_host.resize_array(kps.size()*2);
    int i=0;
    for(const auto& kp:kps){
        locked_keypoints_host.data()[i*2]=kp[0];
        locked_keypoints_host.data()[i*2+1]=kp[1];
        ++i;
    }

    locked_keypoints_device=locked_keypoints_host;
    mhere();

    // mask
    ntp["feature mask"].tic();
    feature_mask(mask_device,locked_keypoints_device,radius());
    mhere();
    ntp["feature mask"].toc();

}
std::vector<cvl::Vector2f>
HarrisDetector::detect(
        const Texture< float, true>& dx,
        const Texture< float, true>& dy,
        const std::vector<cvl::Vector2f>& locked_kps)
{




    mhere();
    ntp["detect total"].tic();
    require(dx.same_size_and_stride(dy), "Must Match");
    if(!dx.same_size_and_stride(corner_scores))
        create_buffers(dx.rows(), dx.cols());

    compute_mask(locked_kps);
    mhere();
    ntp["harris"].tic();
    klt::harris_corner_score( dx, dy, corner_scores);
    mhere();
    ntp["harris"].toc();


    // Perform non-maxima-suppression, should not be on the masked, it creates false peaks...
    ntp["anms"].tic(); // 4ms?
    non_max_supression(corner_scores,
                       corner_scores_extra,
                       radius() );
    mhere();
    ntp["anms"].toc();



    // Multiply min eigenvalue image and mask
    multiply( corner_scores_extra, mask_device, masked_corner_score );



    // Download image to extract feature candidates on CPU
    corner_scores_host=masked_corner_score;//corner_scores;
    mhere();
    ntp["candidates"].tic();
    // Extract feature candidates
    auto cs=candidates(corner_scores_host,
                       min_score());
    ntp["candidates"].toc();
    ntp["detect total"].toc();
  //  std::cout<<ntp<<std::endl;
    return cs;
}
float HarrisDetector::min_score(){
      // parse values from the gui, dont do this inside performance sensitive things...
    update_all();
    return min_score_->value();
}
int HarrisDetector::radius(){
    // parse values from the gui, dont do this inside performance sensitive things...
    update_all();
    return radius_->value();
}

} // end namespace

