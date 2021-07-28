#include <iostream>
#include <kitti/odometry/eval.h>
#include <kitti/odometry/orig_gnu_plot.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/files.h>

using std::cout;using std::endl;
namespace cvl{
namespace kitti{




std::map<int,Result> evaluate(KittiDataset& kd,
                              std::string estimatepath,
                              std::string estimate_name,
                              std::string outputpath    )
{
    std::map<int,std::vector<PoseD>> pcwss;
    for(int seq=0;seq<int(kd.seqs.size());++seq){
        auto ps=readKittiPoses(estimatepath+mlib::toZstring(seq,2)+".txt");
        if(ps.size()>0)
            pcwss[seq]= ps;
    }
    return evaluate(kd,pcwss,estimate_name,outputpath);
}

std::map<int,Result> evaluate(KittiDataset& kd,
                              std::map<int,std::vector<PoseD>> pwcss,
                              std::string estimate_name,
                              std::string outputpath   ){
    std::map<int,Result> results;
    for(auto& [seq, pcws]:pwcss){
        results[seq]=evaluate(kd.seqs[seq], pcws,estimate_name,outputpath);
    }
    return results;
}

Result evaluate(Sequence& seq,
                std::vector<PoseD> Pwcs,
                std::string name,
                std::string outputpath    ){
    std::string op=outputpath+"evaluation_output/classic/";
    mlib::makefilepath(mlib::ensure_dir(op));
    Result result(seq,Pwcs, name);

    plot_sequence(seq.gt_poses(),Pwcs,name,op,seq.name());

    return result;
}










}// end kitti namespace
}// end namespace cvl
