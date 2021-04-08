#include <mlib/utils/string_helpers.h>
#include <mlib/utils/random.h>
#include <mlib/utils/simulator_helpers.h>
#include <mlib/utils/cvl/triangulate.h>
#include <mlib/utils/cvl/triangulate_nl.h>
#include <mlib/utils/random.h>
#include <iomanip>
#include <mlib/utils/mlibtime.h>
 

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace cvl;
using namespace mlib;
using std::cout;using std::vector;using std::endl;

PoseD get_random_pose_which_sees(Vector3d x){
    PoseD P;
    while(true){
        PoseD P=getRandomPose();
        Vector2d y=(P*x).dehom();
        if(!(y.is_in({-1,-1},{1,1}))) continue;
        if((P*x)[2]<0.01) continue;
        return P;
    }
}


class Sample{
public:
    Vector3d x_true, x_trig, x_iter_from_true, x_iter_from_trig, x_ceres;
    std::vector<PoseD> ps;
    std::vector<Vector2d> ys;
    Sample(Vector3d x, std::vector<PoseD> ps,std::vector<Vector2d> ys):x_true(x),ps(ps),ys(ys){
        x_trig=triangulate(ps[0],ps[1],ys[0],ys[1]);
        x_iter_from_true = gn_minimize(ys,ps,x_true);
        x_iter_from_trig = gn_minimize(ys,ps,x_trig);
        x_ceres=triangulate_iterative(ps,ys);

    }
    void estimate(){
        x_trig=triangulate(ps[0],ps[1],ys[0],ys[1]);
        x_iter_from_true = gn_minimize(ys,ps,x_true);
    }
    void estimate_ceres(){
                x_ceres=triangulate_iterative(ps,ys);
    }
    double error(Vector3d xh){
        double err=0;
        if(!xh.isnormal()) return 100000000;
        for(uint i=0;i<ps.size();++i){
            err+=(ys[i]-(ps[i]*xh).dehom()).norm();
        }
        return err/(double)ps.size();
    }

    bool test_estimates(){
        bool check_divergence=false; // divergence tests are problematic because conversions to matrix introduces errors
        if(check_divergence){
            // must not have diverged from x_true
            if(error(x_true)+1e-4<error(x_iter_from_true)){
                cout<< "x_true: "<<x_true << " x_iter_from_true: "<<x_iter_from_true<<endl;
                cout<< error(x_true)<< " "<<error(x_iter_from_true)<< " "<<error(x_true) - error(x_iter_from_true)<<endl;
                return false;
            }

            // must not have diverged from x_trig
            if(error(x_trig)+1e-4<error(x_iter_from_trig))
            {
                cout<< "x_trig: "<<x_true << " x_iter_from_trig: "<<x_iter_from_trig<<endl;
                cout<< error(x_trig)<< " "<<error(x_iter_from_trig)<<" "<<error(x_trig) - error(x_iter_from_trig)<<endl;
                return false;
            }
        }
        // on average, x_iter_from_trig should be better than x_trig how to check?

        if(false){/// must be comparable to ceres
            cout<<"ceres improvement: "<<error(x_true)-error(x_ceres)<<endl;
            cout<<"trig0 improvement: "<<error(x_true)-error(x_trig)<<endl;
            cout<<"trig  improvement: "<<error(x_true)-error(x_iter_from_trig)<<endl;

            // if(error(x_ceres)+ 1e-7*double(ps.size())<error(x_iter_from_true)) return false;
        }
        // should get atleast 90% of ceres improvement

        {
            double cdiff=error(x_true)-error(x_ceres);
            assert(cdiff>=0);
            double idiff=error(x_true)-error(x_iter_from_trig);
            if(idiff*0.9>cdiff) return false;
        }


        return true;
    }

};

std::vector<Sample> get_samples(uint p,uint N, double sigma=1e-2){
    std::vector<Sample> samples;samples.reserve(N);
    for(uint i=0;i<N;++i){
        Vector2d y(randu(-1,1),randu(-1,1));
        Vector3d x=randu(1,100)*(y.homogeneous()); // means most will be far away...
        std::vector<Vector2d> ys;ys.reserve(p);
        std::vector<PoseD> ps;ps.reserve(p);
        for(uint i=0;i<p;++i){
            PoseD P=get_random_pose_which_sees(x);
            ps.push_back(P);
            ys.push_back((P*x).dehom() +  sigma*getRandomUnitVector<double,2>());
        }
        samples.push_back(Sample(x,ps,ys));
    }
    return samples;
}

void simple_test(){
    std::vector<mlib::Timer> timers;
    uint N=1000;
    std::vector<int> poses{2,3,4,5,6,7,20,30,41};
    for(int p:poses){
        // test for stereo
        auto samples=get_samples(p,N);
        double counter=0;
        // build dataset
        timers.push_back(mlib::Timer(toStr(p)));

        for(auto sample:samples){
            //timers.back().tic();            sample.estimate();            timers.back().toc();
            if(!sample.test_estimates())
                counter++;
        }
        //cout<<"p: "<<p<<" has "<<counter<<" failures/1e6"<<endl;
        CHECK(counter<12);
    }
    //cout<<timers<<endl;;
}

TEST_CASE("TRIANGULATE_NL"){
    simple_test();
}
