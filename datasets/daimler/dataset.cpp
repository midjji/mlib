#include <mlib/datasets/daimler/dataset.h>



namespace cvl{




std::vector<std::shared_ptr<StereoSequence>> DaimlerDataset::sequences() const{
    std::vector<std::shared_ptr<StereoSequence>> rets;
    rets.push_back(seq);
    return rets;
};


DaimlerDataset::DaimlerDataset(std::string dataset_path, std::string gt_path):
    seq(std::make_shared<DaimlerSequence>(dataset_path,gt_path)){}
} // end namespace cvl
