
#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>


#include <mlib/datasets/stereo_datasets.h>
#include <mlib/utils/random.h>
#include <mlib/utils/cvl/matrix_adapter.h>
#include <mlib/opencv_util/stereo.h>
#include <unistd.h>

#include <mlib/utils/argparser.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/opencv_util/convert.h>
#include <mlib/opencv_util/imshow.h>
#include <ceres/problem.h>
#include <ceres/loss_function.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/solver.h>
#include <opencv2/features2d.hpp>

using std::cout;
using std::endl;
using namespace cvl;

double gaussian(double mu, double sigma, double val){
    double x=mu-val;

    return std::exp(-(x*x/(sigma*sigma)));
}
double response_v(int row, int col, int target_row, int target_col){
    return gaussian(0,10,(Vector2d(row, col) - Vector2d(target_row, target_col)).norm());
}


cv::Mat1f response(int rows, int cols, int target_row, int target_col){

    cv::Mat1f ret(rows, cols);
    for(int r=0;r<ret.rows;++r)
        for(int c=0;c<ret.cols;++c){
            ret(r,c) = response_v(r,c,target_row, target_col);
        }
    return ret;
}







template<class T> const T& matrix_indexer(const T* const ptr, int row, int col, int stride){
    return ptr[row*stride + col];
}

constexpr int filter_size{31};

class CorrCost
{
public:
    cv::Mat1f image;
    double target_response;
    double row;
    double col;
    double w;

    CorrCost(cv::Mat1f image, double target_response, double  row, double col, double w):
        image(image), target_response(target_response), row(row), col(col),w(w)
    {

    }


    template <typename T>
    bool operator()(
            const T* const filter_param, // just pose works thanks to ceres improvment! // but is quite possibly slower...
            T* residuals) const
    {
        int frows=filter_size;
        int fcols=filter_size;

        T resp=T(0);
        int out_of_image=0;
        for(int r=0;r<frows;++r)
            for(int c=0;c<fcols;++c)
            {
                int r0=r+ row-frows/2;
                int c0=c+ col-fcols/2;
                if(r0<0|| r0>image.rows-1||c0<0||c0>image.cols-1){
                    out_of_image++; continue;
                }
                T cv=T(image(r0,c0)) * matrix_indexer(filter_param, r,c,filter_size);
                resp+=cv;
            }
        residuals[0] = T(w)*(resp -T(target_response));
        return true;
    }
};
ceres::CostFunction*
corr_cost_loss(cv::Mat1f image, double target_response, int  row, int col, double w)
{
    return new ceres::AutoDiffCostFunction<CorrCost, 1, filter_size*filter_size>(new CorrCost(image,  target_response,  row,  col, w));
}



double compute_response(cv::Mat1f image, cv::Mat1f filter, int row, int col)
{
    // assume filter size is odd
    double cost=0;
    int out_of_image=0;
    for(int r=0;r<filter.rows;++r)
        for(int c=0;c<filter.cols;++c)
        {
            int r0=r+ row-filter.rows/2;
            int c0=c+ col-filter.cols/2;
            if(r0<0|| r0>image.rows-1||c0<0||c0>image.cols-1){
                out_of_image++; continue;
            }
            double cv=(image(r0,c0)*filter(r,c));
            cost+=cv;
        }



    return cost;
}

cv::Mat1f compute_response(cv::Mat1f image, cv::Mat1f filter)
{
    cv::Mat1f cost(image.rows, image.cols, 0.0f);
    // attempt to apply the filter to each position
    for(int r=0;r<image.rows;++r)
        for(int c=0;c<image.cols;++c)
        {
            cost(r,c)= compute_response(image, filter, r,c);
        }
    return cost;
}



cv::Mat1f compute_filter(cv::Mat1f image, int row, int col, int size)
{

    ceres::Problem problem;

    std::vector<double> filt;filt.resize(size*size, 1);

    int w=31; // a few extra

    for(int r=row-w;r<row+w+1;r+=1)
        for(int c=col-w;c<col+w+1;c+=1)
        {
            if(r<0||c<0) continue;
            if(r>=image.rows||c>=image.cols) continue;
                double weight=1;
            double dr=r-row;dr*=dr;
            double dc=c-col;dc*=dc;
            double dist=sqrt(dr+dc);
            if(dist>10)
                dist=0.1;
            if(dist>20)
                if(mlib::randu(0,1)<0.9) continue;


            double tr=response_v(r,c,row, col);
            problem.AddResidualBlock(corr_cost_loss(image, tr,r,c, weight),nullptr, &filt[0]);
        }
    for(int i=0;i<10000;++i){
        break;
        int r=row;
        int c=col;
        while(std::abs(r-row)<=w)
            r=mlib::randu(0,image.rows);
        while(std::abs(c-col)<=w)
            c=mlib::randu(0,image.cols);
        double tr=response_v(r,c,row, col);
        problem.AddResidualBlock(corr_cost_loss(image, tr,r,c, 0.1),nullptr, &filt[0]);
    }



    ceres::Solver::Options options;
    {
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.num_threads=std::thread::hardware_concurrency();
        options.max_num_iterations=50;
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout<<summary.FullReport()<<std::endl;

    cv::Mat1f ret(size,size);
    for(int r=0;r<ret.rows;++r)
        for(int c=0;c<ret.cols;++c)
            ret(r,c)=matrix_indexer(&filt[0],r,c,size);

    return ret;
}


using namespace cv;
void match_brief(cv::Mat1b A, cv::Mat1b B)
{
    auto detector=BRISK::create(30,3,1.0f);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(A, noArray(), keypoints1, descriptors1, false);
    detector->detectAndCompute(B, noArray(), keypoints2, descriptors2, false);


        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.1f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        //-- Draw matches
        Mat img_matches;

        drawMatches( A, keypoints1, B, keypoints2, good_matches, img_matches, Scalar::all(-1),
                     Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        imshow("Good Matches", img_matches );
        waitKey(0);
}



int main(int argc, char** argv)
{

    /**
      * Works just fine, but is slow, and limited to less than 40 ish filter size
      * Notably, the window must be very large
      *
      * \note
      * - it does look like not every pixel in the patch must be included in the cost,
      * however, predicting which ones is tricky, looks alot like it will
      *
      *
      *
      * \todo
      *  - switch to dynamic
      *  - add fft2 lib
      *  - use fft2 lib
      *
      *  add multiple features,
      *  extract multiple features
      *  add multiple scales? this is possibly the solution to the size problem
      *
      *
      *
      **/




    auto params = args(argc, argv, { {"dataset", "kitti"},
                                     {"max disparity", "60"},
                                     {"sequence", "0"},
                                     {"offset", "0"},
                       });
    if(params["tracker config"]=="default")
    {
        if(params["dataset"]=="daimler")
            params["tracker config"]="/home/"+std::string(getlogin())+"/co/stereovo/daimler_klt.dat";
        if(params["dataset"]=="kitti")
            params["tracker config"]="/home/"+std::string(getlogin())+"/co/stereovo/kitti_klt.dat";
        if(params["dataset"]=="hilti")
            params["tracker config"]="/home/"+std::string(getlogin())+"/co/stereovo/hilti_klt.dat";
    }

    int sequence=mlib::str2int(params["sequence"]);
    std::cout << "sequence: " << sequence<<"\n";
    int offset=mlib::str2int(params["offset"]);
    std::cout << "offset: " << offset<<"\n";

    auto seq= buffered_kitti_sequence(sequence,offset);
    if(params["dataset"]=="hilti")
        seq= buffered_hilti_sequence(sequence,offset);
    if(params["dataset"]=="daimler")
        seq = buffered_daimler_sequence(offset);

    auto a=seq->next();
    auto b=seq->next();

    match_brief(a->grey1b(), b->grey1b());


    while(true)
    {
auto s=seq->next();

        if(!s) break;
        if(!s->has_stereo()) continue;

        // target
        int tr= 224;
        int tc= 389;



        auto im=s->rgb(0);
        int rows=im.rows;
        int cols=im.cols;
        cvl::imshow("rgb0", s->rgb(0));
        cvl::imshow("rgb1", s->rgb(1));



        cv::Mat3b target=s->rgb(0)(cv::Rect( tc-20, tr -20, 40, 40 ));

        cvl::imshow("Target: ",target);
        cv::Mat1f desired_response=response(rows,cols,tr,tc);

        cvl::imshow("Desired Response: ", image2grey1b(mlib::normalize01(desired_response),255));
        cvl::imshow("Desired Response in Patch: ", desired_response(cv::Rect( tc-20, tr -20, 40, 40 )));



        cv::Mat1f filter = compute_filter(s->grey1f(0), tr,tc, filter_size);
        cv::Mat1f resp0=compute_response(s->grey1f(0), filter);
        cv::Mat1f resp1=compute_response(s->grey1f(1), filter);

        cv::Mat1b sr=image2grey1b(mlib::normalize01(resp0),255);
        cvl::imshow("Source Response: ",sr);
        cv::Mat1b srs=sr(cv::Rect( tc-20, tr -20, 40, 40 ));
        cvl::imshow("Source Response in Patch: ",srs);
        cv::Mat1f diff=(desired_response - resp0);
        mlib::abs_inplace(diff);
        cv::Mat1b diff1b=image2grey1b(mlib::normalize01(diff),255);


        cvl::imshow("Response Error: ",diff1b);
        cvl::imshow("Response Error in Patch: ",diff1b(cv::Rect( tc-20, tr -20, 40, 40 )));
        cvl::imshow("Target Response: ",image2grey1b(mlib::normalize01(resp1),255));
        cvl::imshow("Filter: ", image2grey1b(mlib::normalize01(filter),255));
        cv::waitKey(0);
    }

    return 0;
}

