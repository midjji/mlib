#include <mlib/datasets/daimler/dataset.h>



namespace cvl{




std::vector<std::shared_ptr<StereoSequence>> DaimlerDataset::sequences() const{
    std::vector<std::shared_ptr<StereoSequence>> rets;
    rets.push_back(seq);
    return rets;
};


DaimlerDataset::DaimlerDataset(std::string dataset_path, std::string gt_path):
    seq(DaimlerSequence::create(dataset_path,gt_path)){}


namespace  daimler{

const DaimlerDataset& dataset(std::string path, std::string gt_path)
{
    // magic static, thread safe as of C++11, mostly, always for 17?
    static DaimlerDataset ds(path,gt_path);
    return ds;
}

}



} // end namespace cvl
