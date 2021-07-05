#include <iostream>
#include <kitti/odometry/eval.h>
#include <kitti/odometry/orig_gnu_plot.h>

using std::cout;using std::endl;
namespace cvl{
namespace kitti{

void saferSystemCall(std::string cmd){

    int err=system(cmd.c_str());
    if(err!=0){
        std::ostringstream ss("");
        ss << err;

          throw new std::runtime_error("Failed system call:\n Error: "+ss.str()+"\n "+cmd+"\n");
    }

}

Evaluator::Evaluator(std::string basepath,
                     std::vector<std::string> estimatepaths,
                     std::vector<std::string> names,
                     std::string outputpath){
    kd=KittiDataset(basepath);
    this->estimatepaths=estimatepaths;
    this->names=names;
    this->output_path=outputpath;
}


void Evaluator::init(){
    if(inited) return;
    inited=true;
    cout<<"Initializing dataset"<<endl;
    kd.init();
    cout<<"Reading results"<<endl;
    results.reserve(kd.seqs.size());
    for(Sequence& seq: kd.seqs){

        Result res(seq);
        res.getDistLsh(); // force all to be precomputed! takes a long time to redo!
        results.push_back(res);

    }
    cout<<"Evaluator::init() - done"<<endl;
}

void Evaluator::evaluate(){


    init();
    // ok I have read the results
    // some data is in the sequence, some in the result hmm not ideal!
// for each estimate evaluate the copied result.
    std::string path=output_path+"evaluation_output/classic/";
    std::string cmd="mkdir -p "+path;
    cout<<"cmd"<<cmd<<endl;
    saferSystemCall(cmd);


    for(uint i=0;i<results.size();++i){


        std::vector<PoseD> gts;
        gts=results[i].seq.gt_poses;
        std::vector<std::vector<PoseD>> estimates;estimates.reserve(estimatepaths.size());
        for(const std::string& estimate:estimatepaths){

            Result res=results[i];

            res.init(estimate);

            cout<<res.getDisplayString()<<endl;
            res.evaluate();
            estimates.push_back(res.poses);
        }

        plot_sequence(gts,estimates,names,path,results[i].seq.name());

        // for the ones with gt what is the single worst error

        // what is the single greatest pose change?

        //result.save_evaluation(output_path);// saves the

    }
}


}// end kitti namespace
}// end namespace cvl
