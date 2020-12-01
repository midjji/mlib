#include <mlib/tests/googletest/googletest/include/gtest/gtest.h>




#include <mlib/utils/simulator_helpers.h>
#include <iostream>
#include <mlib/utils/mlibtime.h>
#include <mlib/sfm/p3p/pnp_ransac.h>

#include <mlib/utils/string_helpers.h>





#include <mlib/tests/test_p4p.h>

#include <mlib/sfm/p3p/pnp_ransac.h>



using namespace mlib;
using namespace cvl;
using std::cout;
using std::endl;



class PNP_test{
public:
    virtual PoseD pnp(const std::vector<cvl::Vector3d>& xs,
                      const std::vector<cvl::Vector2d>& yns)=0;
    virtual int getIters()=0;
};


class PNP_lambda: public PNP_test{
public:
    PoseD pnp(const std::vector<cvl::Vector3d>& xs,
              const std::vector<cvl::Vector2d>& yns)
    {
        PnpParams prs;
        cvl::PNP est(xs,yns,prs);

        PoseD pose=est.compute();
        totaliters=est.total_iters;
        return pose;
    }
    int totaliters;
    int getIters(){return totaliters;}
};





void testPnp(PNP_test* pnp){
    std::cerr<<"";
    // generate random poses and pointclouds and test the pnp for them
    std::vector<double> sigmas={0,0.25,0.5,1}; // in pixels

    double outliers=0.5;

    double experiments=1000;

    std::vector<double> failures;failures.resize(sigmas.size(),0);
    std::vector<std::vector<double>> errorss;
    std::vector<std::vector<int>>iterss;

    std::vector<mlib::Timer> timers;timers.resize(sigmas.size());
    for(uint i=0;i<sigmas.size();++i){


        std::vector<double> errors;errors.reserve(experiments);
        std::vector<int> iters;iters.reserve(experiments);
        for(int e=0;e<experiments;++e){
            PointCloudWithNoisyMeasurements data(1000,sigmas[i],outliers);

            timers[i].tic();
            PoseD P=pnp->pnp(data.xs,data.yns);
            EXPECT_TRUE(!std::isnan(P.get4x4().absSum()));
            timers[i].toc();
            iters.push_back(pnp->getIters());


            PoseD I=P*data.Pcw.inverse();
            double error=std::abs(I.getAngle())+I.translation().length();


            errors.push_back(error);

            assert(!std::isnan(error)); // why this works, but not expect true is beyond me...
            //EXPECT_TRUE(!std::isnan(error));
            if((error>0.05)){
                failures[i]++;
            }
        }
        errorss.push_back(errors);
        iterss.push_back(iters);
    }



    std::vector<std::string> headers={"sigma", "bad poses", "ratio", "outliers"};



    headers.push_back("mean err");
    headers.push_back("median err");
    headers.push_back("max err");
    //headers.push_back("mean ms");
    headers.push_back("median ms");
    headers.push_back("max ms");
    headers.push_back("median iters");
    std::vector<std::vector<double>> valss;valss.reserve(sigmas.size());

    assert(errorss.size()==sigmas.size());
    for(uint i=0;i<sigmas.size();++i){
        double mean_ms=timers.at(i).getMean().getMilliSeconds();
        //double median_ms=timers.at(i).getMedian().getMilliSeconds();
        double max_ms=timers.at(i).getMax().getMilliSeconds();

        std::vector<double> vals={ sigmas.at(i),failures.at(i),failures.at(i)/experiments,outliers, mean(errorss[i]),median(errorss[i]),max(errorss[i]),mean_ms,max_ms};
        vals.push_back(median(iterss[i]));
        //vals.push_back(max(iterss[i]));

        valss.push_back(vals);
    }
    cout<<"Experiments:   "<<experiments<<endl;
    cout<<"Outlier ratio: "<<outliers<<endl;
    cout<<displayTable(headers,valss)<<endl;
    for(uint i=0;i<sigmas.size();++i)
        EXPECT_TRUE(experiments<100 || failures.at(i)/experiments<0.05);

}



TEST(PNP_RANSAC,LAMBDA){

    PNP_lambda pnp;
    testPnp(&pnp);
}




int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);    return RUN_ALL_TESTS();

}
