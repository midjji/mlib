#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <kitti/mots/dataset.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/cvl/matrix_adapter.h>

#include <thread>
#include <fstream>
#include <experimental/filesystem>
#include <kitti/mots/calibration.h>

namespace fs = std::experimental::filesystem;
using std::cout;
using std::endl;

namespace cvl{



KittiMotsDataset::KittiMotsDataset(std::string dataset_path){
    if(dataset_path.size()>0)
        if(dataset_path.back()!='/')
            dataset_path.push_back('/');
    path=dataset_path;
}
std::string typepath(bool training){
    if(training) return "training/";
    return "testing/";
}
int KittiMotsDataset::sequences(bool train){
    if(train) return train_sequences;
    return test_sequences;
}
Vector2<uint> KittiMotsDataset::max_train_row_col(){
    uint rowm=0;
    uint colm=0;
    for(auto v:training_row_col){
        rowm=std::max(v[0],rowm);
        colm=std::max(v[1],colm);
    }
    return Vector2<uint>(rowm,colm);
}
void KittiMotsDataset::write_sample_paths(std::string path){
    std::vector<std::string> ls,rs;
    bool training=true;
    std::string base="";
    for(int sequence=0;sequence<train_sequences;++sequence)
        for(uint frameid=0;frameid<train_samples[sequence];++frameid){
            std::string left=base+"images/"+typepath(training)+
                    "image_02/"+
                    mlib::toZstring(sequence,4)+std::string("/")+
                    mlib::toZstring(frameid,6)+std::string(".png");
            std::string right=base+"images/"+typepath(training)+
                    "image_03/"+
                    mlib::toZstring(sequence,4)+"/"+
                    mlib::toZstring(frameid,6)+".png";
            ls.push_back(left);
            rs.push_back(right);
        }
    auto rc=max_train_row_col();
    std::ofstream ofs(path);
    ofs<<rc[0]<< " "<<rc[1]<<endl;
    for(uint i=0;i<ls.size();++i){
        ofs<<ls[i]<< " "<<rs[i]<<"\n";
    }
    ofs.flush();
}
std::shared_ptr<KittiMotsSample>
KittiMotsDataset::get_sample(
        bool training,
        uint sequence,
        uint frameid){

    std::string left=path+"images/"+typepath(training)+
            "image_02/"+
            mlib::toZstring(sequence,4)+std::string("/")+
            mlib::toZstring(frameid,6)+std::string(".png");
    std::string right=path+"images/"+typepath(training)+
            "image_03/"+
            mlib::toZstring(sequence,4)+"/"+
            mlib::toZstring(frameid,6)+".png";
    fs::path lp(left);
    fs::path rp(right);
    if(!fs::exists(lp)){
        cout<<"file not found: "+ left<<endl;
        return nullptr;
    }
    if(!fs::exists(rp)){
        cout<<"file not found: "+ right<<endl;
        return nullptr;
    }
    std::map<std::string, std::string> paths;
    paths["left"]=left;
    paths["right"]=right;
    return std::make_shared<KittiMotsSample>(paths,
                                             training,
                                             sequence,
                                             frameid);
}
int KittiMotsDataset::samples(bool training, uint sequence){
    if(training) return train_samples[sequence];
    return test_samples[sequence];
}
bool image_exists(std::string path,
                  bool training,
                  int sequence,
                  int frameid){

    std::string left=path+"images/"+typepath(training)+
            "image_02/"+
            mlib::toZstring<uint>(sequence,4)+std::string("/")+
            mlib::toZstring<uint>(frameid,6)+std::string(".png");
    return fs::exists(fs::path(left));
}
void KittiMotsDataset::sanity_check_calibrations(){
    /*
    KittiMotsCalibration kmc,kmc0;
    kmc0.read_file(mots_calib_path(path, true, 0));
    for(int i=0;i<train_sequences;++i){
        if(!kmc.read_file(mots_calib_path(path, true, i)))
            cout<<"trouble"<<endl;
        if(kmc.test_common_translations(kmc0)){
            cout<<"trouble"<<endl;

        }

    }
    for(int i=0;i<test_sequences;++i){
        if(!kmc.read_file(mots_calib_path(path, false, i))) cout<<"trouble"<<endl;
        if(kmc.test_common_translations(kmc0))
            cout<<"trouble"<<endl;
    }
*/
}
void KittiMotsDataset::check_count(){
    std::vector<int> trns; trns.resize(train_sequences,0);
    for(int seq=0;seq<train_sequences;++seq){
        int n=0;
        while(image_exists(path,true,seq,n++)){
            trns[seq]++;
        }
    }
    std::vector<int> tsts; tsts.resize(test_sequences,0);
    for(int seq=0;seq<test_sequences;++seq){
        int n=0;
        while(image_exists(path,false,seq,n++)){
            tsts[seq]++;
        }
    }
    cout<<"\ntrain={";
    for(auto t:trns)
        cout<<t<<", ";
    cout<<"}\n";
    cout<<"tsts={";
    for(auto t:tsts)
        cout<<t<<", ";
    cout<<"}\n";
}










int KittiMotsDataset::rows(bool training, int seq){
    if(training) return training_row_col[seq][0];
    return testing_row_col[seq][0];
}
int KittiMotsDataset::cols(bool training, int seq){
    if(training) return training_row_col[seq][1];
    return testing_row_col[seq][1];
}



} // end namespace cvl
