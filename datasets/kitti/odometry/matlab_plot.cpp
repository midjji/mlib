#include <kitti/odometry/matlab_plot.h>
#include <mlib/utils/matlab_helpers.h>
#include <mlib/utils/string_helpers.h>


using std::cout;using std::endl;
using namespace mlib;
namespace cvl{
namespace kitti{
std::vector<double> Result::getTL(std::vector<double> lengths,double thr) const{
    std::vector<double> tl;tl.reserve(kes.size());
    for(double len:lengths){
        double t_err = 0;
        double num   = 0;
        // for all errors do
        for(const KittiError& e:kes)
            if(fabs(e.len-len)<thr){
                t_err += e.t_err;
                ++num;
            }
        // we require at least 3 values
        if (num>2.5)
            tl.push_back(t_err/num);
    }
    return tl;
}
std::vector<double> Result::getRL( std::vector<double> lengths,double thr) const{
    std::vector<double> rl;rl.reserve(kes.size());
    for(double len:lengths){

        double r_err = 0;
        double num   = 0;
        // for all errors do
        for(const KittiError& e:kes)
            if(fabs(e.len-len)<thr){

                r_err += e.r_err;
                ++num;
            }
        // we require at least 3 values
        if (num>2.5)
            rl.push_back(r_err/num);

    }
    return rl;


}
std::vector<double> Result::getTS( std::vector<double> speeds, double thr) const{
    std::vector<double> ts;ts.reserve(kes.size());
    // for each driving speed do (in m/s)
    for (double speed:speeds) {
        double t_err = 0;
        double num   = 0;

        // for all errors do
        for (const KittiError& e:kes)
            if (fabs(e.speed-speed)<thr) {
                t_err += e.t_err;
                ++num;
            }

        // we require at least 3 values
        if (num>2.5)
            ts.push_back(t_err/num);
    }
    return ts;
}
std::vector<double> Result::getRS( std::vector<double> speeds, double thr) const{
    std::vector<double> rs;rs.reserve(kes.size());


    // for each driving speed do (in m/s)
    for (double speed:speeds) {
        double r_err = 0;
        double num   = 0;
        // for all errors do
        for ( const KittiError& e:kes)
            if (fabs(e.speed-speed)<thr) {
                r_err += e.r_err;
                ++num;
            }
        // we require at least 3 values
        if (num>2.5)
            rs.push_back(r_err/num);

    }
    return rs;
}




void tomatlabfile(const std::vector<Result>& results, std::string filename){
    cout<<"tomatlabfile:"<<endl;
    std::vector<double> speeds;speeds.reserve(100);

    for(int i=2;i<25;i+=2)
        speeds.push_back(i);

    std::stringstream ss;
    ss<<"%% matlab file for plotting the kitti error graphs... \n\n";
    ss<<"clear;close all;clc;\n";
    ss<<"speeds="<<getMatlabVector(speeds)<<" % in meters per second\n";
    for(uint i=0;i<results.size();++i)
        ss<<"lengths"+toZstring(i,2)+" = "<<getMatlabVector(results.at(i).lengths)<<" % in meters\n";



    ss<<"legs=[";
    for(uint i=0;i<11;++i){
        ss<<"'Seq: "<<toZstring(i,2)<<"'; ";
    }
    ss<<"]; % legends\n";
    ss<<"cols=[";
    for(uint i=0;i<11;++i){
        ss<<i<<" ";
    }
    ss<<"];\n";

    ss<<"\n";
    double maxtl=0;
    double maxrl=0;
    double maxts=0;
    double maxrs=0;
    for(uint i=0;i<11;++i){
        std::vector<double> tl=results.at(i).getTL(results.at(i).lengths);
        std::vector<double> rl=results.at(i).getRL(results.at(i).lengths);
        std::vector<double> ts=results.at(i).getTS(speeds);
        std::vector<double> rs=results.at(i).getRS(speeds);
        ss<<"tl"<<toZstring(i,2)<<"="<<getMatlabVector(tl)<<"\n";
        ss<<"rl"<<toZstring(i,2)<<"="<<getMatlabVector(rl)<<"\n";
        ss<<"ts"<<toZstring(i,2)<<"="<<getMatlabVector(ts)<<"\n";
        ss<<"rs"<<toZstring(i,2)<<"="<<getMatlabVector(rs)<<"\n";
        ss<<"\n";

        std::vector<double> rld,tld;
        cout<<"getdist"<<endl;

        results.at(i).getDistributions(rld,tld);
        cout<<"getdist for delay 0 "<<endl;

        ss<<"tld"<<toZstring(i,2)<<"="<<getMatlabVector(tld)<<"\n";
        ss<<"rld"<<toZstring(i,2)<<"="<<getMatlabVector(rld)<<"\n";
        maxtl=std::max(max(tl),maxtl);
        maxrl=std::max(max(rl),maxrl);
        maxts=std::max(max(ts),maxts);
        maxrs=std::max(max(rs),maxrs);

    }


    maxtl*=100;
    maxrl*=57.3;
    maxts*=100;
    maxrs*=57.3;

    // actual plotting functions ...


    ss<<"%% plots\n";
    // t/l
    {
        ss<<"figure();\n";
        ss<<"plot(";
        for(uint i=0;i<11;++i){
            std::string chr="*";
            if(i>5)
                chr="s";
            std::string ny="tl"+toZstring(i,2);
            std::string nx="lengths(1:length("+ny+"))";

            ss<<nx<<",100*"<<ny<<",'-"<<chr<<"'";
            if(i!=10)
                ss<<",";
        }
        ss<<");\n";
        ss<<"legend(legs);\n";
        ss<<"title('Translation Error');\n";
        ss<<"xlabel('Path Length [m]');\n";
        ss<<"ylabel('[%%]');\n";
        if(maxtl<2)
            maxtl=2;
        ss<<"ylim([0 "<<maxtl<<"]);\n";
    }

    // r/l
    {
        ss<<"figure();\n";

        ss<<"plot(";
        for(uint i=0;i<11;++i){
            std::string chr="*";
            if(i>5)
                chr="s";
            std::string ny="rl"+toZstring(i,2);
            std::string nx="lengths(1:length("+ny+"))";

            ss<<nx<<",57.3*"<<ny<<",'-"<<chr<<"'";
            if(i!=10)
                ss<<",";
        }
        ss<<");\n";
        ss<<"legend(legs);\n";
        ss<<"title('Rotation Error');\n";
        ss<<"xlabel('Path Length [m]');\n";
        ss<<"ylabel('[deg/m]');\n";

        if(maxrl<0.006)
            maxrl=0.006;
        ss<<"ylim([0 "<<maxrl<<"]);\n";
    }
    // t/s
    {
        ss<<"figure();\n";
        ss<<"plot(";
        for(uint i=0;i<11;++i){
            std::string chr="*";
            if(i>5)
                chr="s";
            std::string ny="ts"+toZstring(i,2);
            std::string nx="speeds(1:length("+ny+"))";

            ss<<"3.6*"<<nx<<",100*"<<ny<<",'-"<<chr<<"'";
            if(i!=10)
                ss<<",";
        }
        ss<<");\n";
        ss<<"legend(legs);\n";
        ss<<"title('Translation Error');\n";
        ss<<"xlabel('Speed [km/h]');\n";
        ss<<"ylabel('[%%]');\n";
        if(maxts<2)
            maxts=2;
        ss<<"ylim([0 "<<maxts<<"]);\n";


    }
    //r/s
    {
        ss<<"figure();\n";
        ss<<"plot(";
        for(uint i=0;i<11;++i){
            std::string chr="*";
            if(i>5)
                chr="s";
            std::string ny="rs"+toZstring(i,2);
            std::string nx="speeds(1:length("+ny+"))";

            ss<<"3.6*"<<nx<<",57.3*"<<ny<<",'-"<<chr<<"'";
            if(i!=10)
                ss<<",";
        }
        ss<<");\n";
        ss<<"legend(legs);\n";
        ss<<"title('Rotation Error');\n";
        ss<<"xlabel('Speed [km/h]');\n";
        ss<<"ylabel('[deg/m]');\n";
        if(maxrs<0.006)
            maxrs=0.006;
        ss<<"ylim([0 "<<maxrs<<"]);\n";
    }


    // tl dist
    {
        ss<<"figure();\n";
        ss<<"yt=[";
        for(uint i=0;i<11;++i){
            std::string ny="tld"+toZstring(i,2);
            ss<<ny<<"' ";
        }
        ss<<"];\n";
        ss<<"[a bins]=hist(yt,25);\n";
        for(uint i=0;i<11;++i){
            std::string ny="tld"+toZstring(i,2);
            std::string nyh=ny+"h";
            ss<<"["<<nyh<<" b]=hist("<<ny<<",bins);\n";
            ss<<nyh<<"="<<nyh<<"./(sum("<<nyh<<"));\n";
        }
        for(uint i=0;i<11;++i){
            std::string ny="tld"+toZstring(i,2);
            std::string nyh=ny+"h";
            ss<<nyh<<" = log("<<nyh<<");\n";
        }

        ss<<"plot(";
        for(uint i=0;i<11;++i){
            std::string chr="*";
            if(i>5)
                chr="s";
            std::string ny="tld"+toZstring(i,2);
            std::string nyh=ny+"h";
            ss<<"bins,"<<nyh<<",'-"<<chr<<"'";
            if(i!=10)
                ss<<",";
        }
        ss<<");\n";
        ss<<"legend(legs);\n";
        ss<<"title('Translation Error Distribution');\n";
        ss<<"xlabel('Bins [cm]');\n";
        ss<<"ylabel('Log Likelihood');\n";
    }
    // rl dist
    {
        ss<<"figure();\n";
        ss<<"yt=[";
        for(uint i=0;i<11;++i){
            std::string ny="rld"+toZstring(i,2);
            ss<<ny<<"' ";
        }
        ss<<"];\n";
        ss<<"[a bins]=hist(yt,25);\n";
        for(uint i=0;i<11;++i){
            std::string ny="rld"+toZstring(i,2);
            std::string nyh=ny+"h";
            ss<<"["<<nyh<<" b]=hist("<<ny<<",bins);\n";
            ss<<nyh<<"="<<nyh<<"./(sum("<<nyh<<"));\n";
        }
        for(uint i=0;i<11;++i){
            std::string ny="rld"+toZstring(i,2);
            std::string nyh=ny+"h";
            ss<<nyh<<" = log("<<nyh<<");\n";
        }

        ss<<"plot(";
        for(uint i=0;i<11;++i){
            std::string chr="*";
            if(i>5)
                chr="s";
            std::string ny="rld"+toZstring(i,2);
            std::string nyh=ny+"h";
            ss<<"bins,"<<nyh<<",'-"<<chr<<"'";
            if(i!=10)
                ss<<",";
        }
        ss<<");\n";
        ss<<"legend(legs);\n";
        ss<<"title('Rotation Error Distribution');\n";
        ss<<"xlabel('Bins [degrees]');\n";
        ss<<"ylabel('Log Likelihood');\n";
    }







    cout<<"filename: "<<filename<<endl;
    std::ofstream of; of.open(filename); of<<ss.str()<<endl; of.close();
}


}// end kitti namespace
}// end namespace cvl
