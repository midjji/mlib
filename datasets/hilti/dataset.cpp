#include <mlib/datasets/hilti/dataset.h>
#include <mlib/utils/files.h>


namespace cvl
{
namespace hilti {


std::shared_ptr<Sequence> Dataset::sequence(int index) const{

    auto it=seqs.find(index);
    if(it==seqs.end()) {mlog()<<index<<"\n";wtf();}
    return it->second;
}
Dataset::Dataset(std::string dataset_path)
{
    dataset_path=mlib::ensure_dir(dataset_path);

    // dataset format.
    //hilti/preprocessed/sequence_name/
    //                                 times.txt // all the times around
    //                                 post_rectification_calibration.txt
    //                                 left/{time}.exr   // left stereo rectified, not just nonlinear rectified!
    //                                 right/{time}.exr   // left stereo rectified, not just nonlinear rectified!
    //                                 cam0 symlink to left, ie we swap them
    //                                 cam1 symlink to right
    //                                 cam2/{time}.exr
    //                                 cam3/{time}.exr
    //                                 cam4/{time}.exr
    //                                 disparity/ // this is a symlink to a selected disparity method folder
    //                                 disparity_method0/{time}.exr





    for(const auto& [num,seq]:num2sequence)
        seqs[num]=Sequence::create(dataset_path+seq+"/",seq);

}

const Dataset& dataset(std::string path)
{
    // magic static, thread safe as of C++11, mostly, always for 17?
    static Dataset ds(path);
    return ds;
}

}

} // end namespace cvl
