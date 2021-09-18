#include <mlib/datasets/hilti/dataset.h>
#include <mlib/utils/files.h>


namespace cvl
{
namespace hilti {



Dataset::Dataset(std::string dataset_path)
{

    for(auto [seq_path, training]:sequence_names_){
        seqs.push_back(Sequence(mlib::ensure_dir(dataset_path)+seq_path+"/"));
    }
}


}
} // end namespace cvl
