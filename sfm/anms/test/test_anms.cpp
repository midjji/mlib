#include <iostream>

#include <mlib/utils/simulator_helpers.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/sfm/anms/base.h>
#include <mlib/sfm/anms/draw.h>
#include <mlib/sfm/anms/grid.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using namespace mlib;
using std::endl;
using std::cout;
using namespace cvl;

// anms requires two properties to work
// if templated with a thin wrapping class or through inheritance?




int id=0;
anms::Data getRandomData(){
    return anms::Data(mlib::randu<double>(-10,10),mlib::randu<double>(0,1280),mlib::randu<double>(0,800),id++);
}

std::vector<anms::Data> getRandomDatas(int Clusters,int per_cluster){
    std::vector<Vector2f> clusters;clusters.reserve(Clusters);
    // clusters span the whole image
    for(int i=0;i<Clusters;++i)
        clusters.push_back(Vector2f(mlib::randu<double>(100,1180),mlib::randu<double>(100,700)));

    // samples span +- 30 of the cluster

    std::vector<anms::Data> datas;datas.reserve(Clusters*per_cluster);
    for(Vector2f cluster:clusters)
        for(int i=0;i<per_cluster;++i){
            datas.push_back(anms::Data(mlib::randui(-10,10),mlib::randn<double>(cluster[0],15),mlib::randn<double>(cluster[1],15),id++));
        }
    // show(datas);
    return datas;
}
std::vector<anms::Data> getRandomDatas(uint N){
    auto ds=getRandomDatas(sqrt(N),sqrt(N));
    while(ds.size()<N)
        ds.push_back(getRandomData());
    return ds;
}

void roundCoordinates(std::vector<anms::Data>& datas){
    for(auto& data:datas){
        data.y[0]=std::round(data.y[0]);
        data.y[1]=std::round(data.y[1]);
    }
}






bool check_distance(std::vector<anms::Data>& datas,double radius){
    for(uint i=0;i<datas.size();++i)
        for(uint j=0;j<datas.size();++j)
            if(i!=j){
                if((datas[i].y-datas[j].y).length()<radius) return false;
            }
    return true;
}












TEST_CASE("ANMS,BASE_IMPLEMENTATION"){
    // effect on a regular grid of even points is predictable
    std::vector<anms::Data> grid;grid.reserve(100);
    std::vector<int> ids;ids.reserve(100);

    for(int i=0;i<10;++i)
        for(int j=0;j<10;++j){
            grid.push_back(anms::Data(i*10+j,i,j,i*10+ j));
            ids.push_back(i*10+j);

        }


    mlib::random::shuffle(grid);

    // test 1 apply anms to shuffled data where none should be removed
    {
        anms::Solver sol;
        sol.init(grid);
        sol.compute(0.5,-1); // should have no effect, closest one is at distance 1


        CHECK(sol.filtered.size()==grid.size());
        // verify all ids are left
        auto filtered_ids=getIds(sol.filtered);
        sort(filtered_ids.begin(),filtered_ids.end());
        sort(ids.begin(),ids.end());
        CHECK(ids.size()==filtered_ids.size());
        for(uint i=0;i<ids.size();++i)
            CHECK(ids[i]==filtered_ids[i]);
    }


    {
        // add some random bad ones that should be removed
        // all between the old
        auto grid2=grid;grid2.reserve(grid2.size()*3);
        for(int i=0;i<100;++i){
            grid2.push_back(anms::Data(-1,randu<double>(1,8),randu<double>(1,8),101+i));
        }
        random::shuffle(grid2);

        anms::Solver sol;
        sol.init(grid2);
        sol.compute(0.71,-1); // greater than 1/sqrt(2) which is the minimum distance they can be at

        CHECK(sol.filtered.size()==grid.size());
        // verify all ids are left
        auto filtered_ids=getIds(sol.filtered);
        sort(filtered_ids.begin(),filtered_ids.end());
        sort(ids.begin(),ids.end());

        CHECK(ids.size()==filtered_ids.size());
        for(uint i=0;i<ids.size();++i)
            CHECK(ids[i]==filtered_ids[i]);
    }
}

bool matching_ids(std::vector<int> as, std::vector<int> bs){
    if(as.size()!=bs.size()) return false;
    sort(as.begin(),as.end());
    sort(bs.begin(),bs.end());
    for(uint i=0;i<as.size();++i)
        if(as[i]!=bs[i]) return false;
    return true;
}
bool matching_ids(std::vector<anms::Data> as,
                  std::vector<anms::Data> bs){
    if(as.size()!=bs.size()) return false;
    return matching_ids(getIds(as),getIds(bs));
}

void time_anms(anms::Solver* solver, bool pixel_accurate=true){
    anms::Solver gt;
    int experiments=10;
    std::vector<uint> sizes={500,1000,10000};

    double radius=25;

    std::vector<mlib::Timer> inits;inits.resize(sizes.size());
    std::vector<mlib::Timer> timers;timers.resize(sizes.size());
    std::vector<double> remaining;remaining.resize(sizes.size());
    for(uint size=0;size<sizes.size();++size){
        for(int i=0;i<experiments;++i){


            auto ds=getRandomDatas(sizes[size]); // between 0,100, 0,100 => max 10k at distance 1
            if(pixel_accurate)
                roundCoordinates(ds);
            CHECK(ds.size()==sizes[size]);

            timers[size].tic();
            inits[size].tic();
            solver->init(ds);
            inits[size].toc();
            solver->compute(radius,-1);
            timers[size].toc();
            remaining[size]=solver->filtered.size();
            CHECK(check_distance(solver->filtered,radius));
            CHECK(solver->filtered.size()<=ds.size());
            CHECK(solver->filtered.size()>0);

            // verify all ids are left
            if(solver->exact()){
                gt.init(ds);
                gt.compute(radius,-1);
                CHECK(gt.filtered.size()==solver->filtered.size());
                CHECK(matching_ids(gt.filtered,solver->filtered));
            }
        }
    }
    cout<<"Experiments:   "<<experiments<<endl;
    std::vector<std::string> headers={"Size", "remaining"};
    headers.push_back("init mean ms");
    headers.push_back("mean ms");
    headers.push_back("median ms");
    headers.push_back("max ms");
    std::vector<std::vector<double>> rows;
    for(uint i=0;i<sizes.size();++i){
        std::vector<double> row;row.reserve(64);
        row.push_back(sizes[i]);
        row.push_back(remaining[i]);
        row.push_back(inits[i].getMean().getMilliSeconds());
        row.push_back(timers[i].getMean().getMilliSeconds());
        row.push_back(timers[i].getMedian().getMilliSeconds());
        row.push_back(timers[i].getMax().getMilliSeconds());
        rows.push_back(row);
    }
    cout<<displayTable(headers,rows)<<endl;
}
void test_anms(anms::Solver* solver, bool pixel_accurate=true){
    anms::Solver gt;
    int experiments=1;
    std::vector<uint> sizes={1000};
    double radius=25;
    std::vector<double> remaining;remaining.resize(sizes.size());
    for(uint size=0;size<sizes.size();++size){
        for(int i=0;i<experiments;++i){


            auto ds=getRandomDatas(sizes[size]); // between 0,100, 0,100 => max 10k at distance 1
            if(pixel_accurate)
                roundCoordinates(ds);
            CHECK(ds.size()==sizes[size]);


            solver->init(ds);

            solver->compute(radius,-1);

            remaining[size]=solver->filtered.size();
            CHECK(check_distance(solver->filtered,radius));
            CHECK(solver->filtered.size()<=ds.size());
            CHECK(solver->filtered.size()>0);

            // verify all ids are left
            if(solver->exact())
            {
                gt.init(ds);
                gt.compute(radius,-1);
                CHECK(gt.filtered.size()==solver->filtered.size());
                CHECK(matching_ids(gt.filtered,solver->filtered));
            }
        }
    }
}

TEST_CASE("ANMS,BASIC_EXACT"){

  anms::Solver solver;
    test_anms(&solver);
}

TEST_CASE("ANMS,GRID_EXACT"){

  anms::GridSolver solver;
    test_anms(&solver);
}

TEST_CASE("AANMS,DRAW_METHOD"){
    anms::DrawSolver solver(1280,1280,1000);
    test_anms(&solver);
}




