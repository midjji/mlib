#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <thread>
#include <fstream>
#include <experimental/filesystem>
#include <sqlite_orm.h>
#include <daimler/dataset.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/cvl/matrix_adapter.h>


namespace fs = std::experimental::filesystem;
using std::cout;
using std::endl;
using namespace sqlite_orm;
namespace cvl{


int DaimlerDataset::samples() const{return total_samples;}
double DaimlerDataset::time_to_frameid_factor() const{return 33;}

std::string parse(std::string path2, std::string gt_path){

    if(gt_path=="")
        return path2+"database/boundingboxes.sqlite";
    return gt_path;
}

DaimlerDataset::DaimlerDataset(std::string path2, std::string gt_path):path(path2),
  gt_storage(make_storage(parse(path2, gt_path).c_str(),
                          make_table("boundingboxes",
                                     make_column("id", &mtable::GTRow::uid, autoincrement(), primary_key()),
                                     make_column("frame_id", &mtable::GTRow::frame_id),
                                     make_column("imo_id", &mtable::GTRow::imo_id),
                                     make_column("row_start", &mtable::GTRow::row_start),
                                     make_column("col_start", &mtable::GTRow::col_start),
                                     make_column("row_end", &mtable::GTRow::row_end),
                                     make_column("col_end", &mtable::GTRow::col_end),
                                     make_column("x", &mtable::GTRow::x),
                                     make_column("y", &mtable::GTRow::y),
                                     make_column("z", &mtable::GTRow::z)))){

    cout<<"created daimler dataset"<<endl;
    if(path.size()> 0 && path.back()!='/')
        path.push_back('/');
    // read metadata



    std::ifstream ifs(path+"metadata.txt");
    cout<<"path: "<<path<<endl;
    assert(ifs);
    ifs>>total_samples;



    // verify the dataset is there!
    if(!get_sample(0)){assert("not found" && false); throw new std::runtime_error("dataset missing");}
    cout<<"created daimler dataset done"<<endl;
}


cv::Mat1w get_bw_labels(cv::Mat1b bw)
{

    {
        cv::namedWindow("bw",cv::WINDOW_GUI_EXPANDED);
        cv::imshow("bw", bw);

    }

    std::vector<int> ls; ls.reserve(bw.rows*bw.cols);
    for(int i=0;i<bw.rows*bw.cols;++i) ls.push_back(i);

    std::vector<int*> lsp; lsp.reserve(bw.rows*bw.cols);
    for(int i=0;i<bw.rows*bw.cols;++i) lsp.push_back(&ls[0]);




    cvl::MatrixAdapter<int*> tmp(lsp.data(), bw.rows, bw.cols);
    int current=1;

    // pass one, this is parallell over rows, given atomic current, or row uniq ptrs...
    for(int r=0;r<bw.rows;++r){
        for(int c=0;c<bw.cols;++c){
            if(bw(r,c)>0){

                // if left is has label, than use that, else if zero then set to next available id
                if(c==0 || (bw(r,c-1)==0) )
                    tmp(r,c)=&(ls[current++]);
                else
                    tmp(r,c)=tmp(r,c-1);
                if(*tmp(r,c)==0)
                    cout<<"set to: "<<*tmp(r,c)<<endl;
            }
        }
    }
    // intermediate result
    {
        cv::Mat3b labels(bw.rows, bw.cols);
        for(int r=0;r<bw.rows;++r)
            for(int c=0;c<bw.cols;++c)
                labels(r,c)=cv::Vec3b(0,uchar(*tmp(r,c)/256),uchar(*tmp(r,c)));
        cv::namedWindow("labels1",cv::WINDOW_GUI_EXPANDED);
        cv::imshow("labels1", labels);
        cv::waitKey(0);
    }



    // not so parallell...
    for(int r=1;r<bw.rows;++r){
        for(int c=0;c<bw.cols;++c){
            //cout<<"r,c"<<r<<" "<<c<<endl;
            if((bw(r,c)>0) && (bw(r-1,c)>0)){
                *tmp(r,c)=*tmp(r-1,c);
            }
        }
    }
    // intermediate result
    {
        cv::Mat3b labels(bw.rows, bw.cols);
        for(int r=0;r<bw.rows;++r)
            for(int c=0;c<bw.cols;++c)
                labels(r,c)=cv::Vec3b(0,uint8_t(*tmp(r,c)/256),uint8_t(*tmp(r,c)));
        cv::namedWindow("labels2",cv::WINDOW_GUI_EXPANDED);
        cv::imshow("labels2", labels);
        cv::waitKey(0);
    }
    // not so parallell...
    for(int r=bw.rows-2;r>0;--r){
        for(int c=0;c<bw.cols;++c){
            //cout<<"r,c"<<r<<" "<<c<<endl;
            //cout<<"rows, cols: "<<bw.rows<<" "<<bw.cols<<endl;
            if((bw(r,c)>0) && (bw(r+1,c)>0)){
                *tmp(r,c)=*tmp(r+1,c);
            }
        }
    }
    // intermediate result
    {
        cv::Mat3b labels(bw.rows, bw.cols);
        for(int r=0;r<bw.rows;++r)
            for(int c=0;c<bw.cols;++c)
                labels(r,c)=cv::Vec3b(0,uchar(*tmp(r,c)/256),uchar(*tmp(r,c)));
        cv::namedWindow("labels3",cv::WINDOW_GUI_EXPANDED);
        cv::imshow("labels3", labels);
        cv::waitKey(0);
    }
    // think that workes... nope probably not...
    // now go from labels to min labels...




    // convert to labels...
    cv::Mat1w labels(bw.rows, bw.cols);
    std::set<int> ids;
    for(int r=0;r<bw.rows;++r){
        for(int c=0;c<bw.cols;++c){
            labels(r,c)=*tmp(r,c);
            ids.insert(*tmp(r,c));
        }
    }
    std::vector<int> vids;vids.reserve(ids.size());
    for(int id:ids) vids.push_back(id);
    std::map<int,int> map;
    for(uint i=0;i<vids.size();++i)
        map[vids[i]]=i;


    for(int r=0;r<bw.rows;++r)
        for(int c=0;c<bw.cols;++c)
            labels(r,c)=map[labels(r,c)];

    //if(labels(r,c))





    return labels;
}

cv::Mat3b convert2rgb8(cv::Mat img){
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c)
            rgb(r,c)=cv::Vec3b(img.at<unsigned short>(r,c)/(256*256),img.at<unsigned short>(r,c)/256,img.at<unsigned short>(r,c));
    return rgb;
}

cv::Mat1b DaimlerDataset::get_cars(cv::Mat1b labels){
    // so the car number is 16
    // the truck one is 18

    //Now lets give each car a unique number ...
    cv::Mat1b cars(labels.rows,labels.cols, uint8_t(1));
    //cars=cars==1;
    cars=(labels==16) + (labels==18);
    return cars;
}

bool test_im(cv::Mat im){
    if(im.data==nullptr) return false;
    if(im.rows==0) return false;
    if(im.cols==0) return false;
    return true;
}

float loc_win_val(cv::Mat1f& im, int row, int col){
    for(int r=-1;r<2;++r)
        for(int c=-1;c<col+2;++c){
            float v=im(r+row,c+col);
            if(v>=0) return v;
        }
    return im(row,col);
}

void post_process_disparity(cv::Mat1f& disparity){
    // there are a bit too many empty ones... so use a fill by nearest to shrink the holes
    // actually a really crappy fill is sufficient!
    for(int r=0;r<disparity.rows;++r)
        for(int c=0;c<disparity.cols;++c)
            if(disparity(r,c)<0|| disparity(r,c)>256)
                disparity(r,c)=-1;

    // grow really...
    for(int i=0;i<2;++i){
        cv::Mat1f disp=disparity.clone();
        // grow 1 from the side
        for(int r=10;r<disp.rows-10;++r){
            for(int c=10;c<disp.cols-10;++c){
                if(disp(r,c)>=0) continue;
                float f=loc_win_val(disparity,r,c);
                if(f>=0)
                    disp(r,c)=f;
            }
        }
        disparity=disp;
    }
}

bool DaimlerDataset::read_images(uint sample_index,
                                 cv::Mat1w& left,
                                 cv::Mat1w& right,
                                 cv::Mat1b& labels,
                                 cv::Mat1f& disparity)
{

    std::vector<std::thread> thrs;thrs.reserve(4);
    thrs.push_back(std::thread([&]() {
        left=       cv::imread(path+"left/"     + mlib::toZstring(sample_index,6)+".png",cv::IMREAD_ANYDEPTH);
        if(!test_im(left))
            cout<<"failed left: \""<<path+"left/"     + mlib::toZstring(sample_index,6)+".png\""<<endl;
    }));
    thrs.push_back(std::thread([&]() {
        if(read_right)
            right=      cv::imread(path+"right/"    + mlib::toZstring(sample_index,6)+".png",cv::IMREAD_ANYDEPTH);
        else
            right=cv::Mat1w(10,10);

    }));
    thrs.push_back(std::thread([&]() {
        disparity=  cv::imread(path+"lstereo/"  + mlib::toZstring(sample_index,6)+".exr",cv::IMREAD_ANYDEPTH);

        //post_process_disparity(disparity);

        //cout<<ppdt<<endl;
    }));

    thrs.push_back(std::thread([&]() {
        labels=     cv::imread(path+"labels/"   + mlib::toZstring(sample_index,6)+".png",cv::IMREAD_ANYDEPTH);
        labels=get_cars(labels);
    }));
    for(auto& thr:thrs) thr.join();
    return (test_im(left) && test_im(right) && test_im(disparity) && test_im(labels));
}


std::shared_ptr<DaimlerSample> DaimlerDataset::get_sample(uint index)
{

    cv::Mat1w left, right;
    cv::Mat1b cars;
    cv::Mat1f disparity;
    if(!read_images(index,left,right, cars, disparity)) return nullptr;



    std::vector<cv::Mat1w> images;
    images.push_back(left);
    images.push_back(right);
    // imo_id, boundingbox

    return std::make_shared<DaimlerSample>(images,disparity,cars,index);
}






} // end namespace cvl
