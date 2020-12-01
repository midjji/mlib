#if 0
#include <mlib/cuda/pnp.h>
#include <mlib/utils/random.h>
#include <mlib/sfm/p3p/pnp_ransac.h>
#include <mlib/utils/vector.h>
using std::cout;
using std::endl;


namespace cvl {
namespace cuda {

__host__ __device__ void round6(double& d){d*=1e6;int i=d;d=i;d*=1e-6;}
template<int THREADS>
__global__
/**
 * @brief getInliers
 * @param xs
 * @param yns
 * @param disps
 * @param poses
 * @param inliers
 *
 * 32 threads per block,
 * one block per pose ~1000 blocks
 */
void getInliers2(MatrixAdapter<Vector3d> xs,
                 MatrixAdapter<Vector2d> yns,
                 MatrixAdapter<double> disps,
                 MatrixAdapter<PoseD> poses,
                 MatrixAdapter<int> inlierss,
                 double baseline,
                 double thr){
    assert(xs.rows==1);
    assert(yns.rows==1);
    assert(disps.rows==1);
    assert(inlierss.rows==1);
    assert(poses.rows==1);
    assert(xs.cols==yns.cols);
    assert(disps.cols==yns.cols);
    //printf("%i %i\n", poses.cols,inlierss.cols);
    assert(poses.cols==inlierss.cols);



    PoseD p=poses(0,blockIdx.x);
    Matrix4d P=p.get4x4();



    __syncthreads();
    int inlier=0;
    for(int i=0;i<(xs.cols+THREADS-1)/THREADS;++i){
        int index=THREADS*i+threadIdx.x;
        if(index<xs.cols){
            cvl::Vector3d X=xs(0,index);
            cvl::Vector3d Xl=(P*X.homogeneous()).dehom();


            double x=Xl[0];
            double y=Xl[1];
            double z=Xl[2];

            double iz=1.0/z;
            cvl::Vector2d yn=yns(0,index);
            double err1=x *iz - yn(0);
            double err2=y *iz - yn(1);
            double disp=disps(0,index);
            double err3=(disp - baseline *iz);

            double error=err1*err1 + err2*err2 +err3*err3;
            round6(error);
            inlier += (error < thr) ? 1 : 0;
        }
        //if(N-i +inliers<best_inliers) break;
        //mle += std::min(errors[i],thr);
    }
    __shared__ int inliers[THREADS];
    inliers[threadIdx.x]=inlier;
    __syncthreads();
    if(threadIdx.x==0){
        int inl=0;
        for(int i=0;i<THREADS;++i)
            inl+=inliers[i];
        inlierss(0,blockIdx.x)=inl;
    }
}

__global__
/**
 * @brief getInliers
 * @param xs
 * @param yns
 * @param disps
 * @param poses
 * @param inliers
 *
 * 32 threads per block,
 * one block per pose ~1000 blocks
 */
void getInliers(MatrixAdapter<Vector3d> xs,
                MatrixAdapter<Vector2d> yns,
                MatrixAdapter<double> disps,
                MatrixAdapter<PoseD> poses,
                MatrixAdapter<int> inlierss,
                double baseline,
                double thr){
    assert(xs.rows==1);
    assert(yns.rows==1);
    assert(disps.rows==1);
    assert(inlierss.rows==1);
    assert(poses.rows==1);
    assert(xs.cols==yns.cols);
    assert(disps.cols==yns.cols);
    assert(poses.cols==inlierss.cols);

    PoseD P=poses(0,blockIdx.x);

    // synch here ? nah
    if(threadIdx.x==0){
        int inliers=0;
        for(int i=0;i<xs.cols;++i){

            cvl::Vector3d Xl=P*xs(0,i);


            double x=Xl[0];
            double y=Xl[1];
            double z=Xl[2];

            //assert(disps[i]>0);
            double iz=1.0/z;
            cvl::Vector2d yn=yns(0,i);
            double err1=x *iz - yn(0);
            double err2=y *iz - yn(1);
            double disp=disps(0,i);
            double err3=(disp - baseline *iz);

            double error=err1*err1 + err2*err2 +err3*err3;
            round6(error);
            inliers += (error < thr) ? 1 : 0;

            //if(N-i +inliers<best_inliers) break;
            //mle += std::min(errors[i],thr);
        }
        inlierss(0,blockIdx.x)=inliers;
    }


}







void PNPC::init(const std::vector<cvl::Vector3d>& xs,
                const std::vector<cvl::Vector2d>& yns,
                const std::vector<double>& disps,
                double baseline){
    assert(xs.size()==yns.size());
    assert(disps.size()==yns.size());
    cudaFree(0);
    dev_poses       =dmm.allocate<cvl::PoseD>(1,N);
    dev_inlierss    =dmm.allocate<int>(1,N);

    dev_xs      =dmm.upload(xs);
    dev_yns     =dmm.upload(yns);
    dev_disps   =dmm.upload(disps);

    this->xs=xs;
    this->yns=yns;
    this->disps=disps;
    this->baseline=baseline;
    this->thr=0.004;

}





void getError(const Matrix4d& M,
              std::set<int> indexes,
              const std::vector<cvl::Vector3d>& xs,
              const std::vector<cvl::Vector2d>& yns,
              const std::vector<double>& disps,
              double baseline,
              std::vector<double>& errors){
    assert(yns.size()==xs.size());

    assert(yns.size()==disps.size());
    errors.clear();errors.resize(indexes.size());
int n=0;
    for(int i:indexes) {

        cvl::Vector4d X=xs[i].homogeneous();
        Vector4d Xl=M*X;
        //Vector3d XR=pose*xs[i]; is x10 slower!



        double x=Xl[0];
        double y=Xl[1];
        double z=Xl[2];

        //assert(disps[i]>0);
        double iz=1.0/z;
        double err1=x *iz - yns[i](0);
        double err2=y *iz - yns[i](1);
        double err3=(disps[i] - baseline *iz);

        errors[n]=err1*err1 + err2*err2 +err3*err3;
    n++;
    }
}

void PNPC::compute(){
    thr*=thr;
    // assume a certain number of iterations required say 1000.
    // compute 1000 poses, should take 1.5ms on cpu

    std::vector<PoseD> poses;poses.reserve(N);
    std::set<int> indexes;
    std::vector<double> errors;
    for(int i=0;i<N*100 && poses.size()<N;++i){
        PoseD pose;
        mlib::random::getNUnique<int>(4,xs.size()-1,indexes);
        if(mlib::klas::p4p( xs,yns,indexes,pose)){
            getError(pose.get4x4(),indexes,xs,yns,disps,baseline,errors);
            if(mlib::sum(errors)<thr) poses.push_back(pose);
        }
    }

    // prefilter the poses given the egomotion prior

    // upload the poses

    dmm.upload(poses,dev_poses);
    std::vector<int> inlierss;inlierss.resize(poses.size(),0);
    dmm.upload(inlierss,dev_inlierss);
    // compute the errors
    dim3 grid(poses.size(),1);
    dim3 threads(32,1,1);
    getInliers2<32><<<grid,threads,0,dmm.pool.stream(0)>>>(dev_xs,dev_yns,dev_disps,dev_poses,dev_inlierss,baseline,thr);
    dmm.pool.synchronize(0);
    inlierss=dmm.download2vector(dev_inlierss);

    if(true){
        getInliers<<<grid,threads,0,dmm.pool.stream(0)>>>(dev_xs,dev_yns,dev_disps,dev_poses,dev_inlierss,baseline,thr);
        dmm.pool.synchronize(0);
        std::vector<int> inliersstmp;
        inliersstmp=dmm.download2vector(dev_inlierss);


        assert(inliersstmp.size()==inlierss.size());

        for(int i=0;i<inlierss.size();++i){
            if(inlierss[i]!=inliersstmp[i])
            cout<<"inliers: "<<inlierss[i]<<" "<<inliersstmp[i]<<" "<<i<<endl;
            assert(inliersstmp[i]==inlierss[i]);
        }
    }


    int best=0;
    int index=0;
    for(int i=0;i<poses.size();++i){
        // cout<<"best: "<<best<<endl;
        // cout<<"index: "<<index<<endl;
        if(inlierss[i]>best){ index=i; best=inlierss[i];}
    }
    //cout<<"best: "<<best<<endl;
    //cout<<"index: "<<index<<endl;
    bestPose=poses[index];

    // select the max

    // refine the result...


}

} // end cuda
} // end cvl
#endif
