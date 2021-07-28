#include <kitti/odometry/orig_gnu_plot.h>
#include <sstream>
#include <kitti/odometry/eval.h>
#include <mlib/utils/vector.h>
#include <mlib/utils/files.h>
#include <mlib/utils/sys.h>
#include <fstream>

using std::cout;using std::endl;
namespace cvl{
namespace kitti{


void write_error_data(std::vector<KittiError>& es, std::string output_path, std::string name,std::vector<double> lengths){


    std::stringstream tl,rl,ts,rs;
    // for each segment length do
    for(double len:lengths){
        double t_err = 0;
        double r_err = 0;
        double num   = 0;
        // for all errors do
        for(KittiError& e:es)
            if(fabs(e.len-len)<1.0){
                t_err += e.t_err;
                r_err += e.r_err;
                ++num;
            }
        // we require at least 3 values
        if (num>2.5) {
            tl<<len<<" "<<t_err/num<<"\n";
            rl<<len<<" "<<r_err/num<<"\n";
        }
    }
    // for each driving speed do (in m/s)
    for (int speed=2; speed<25; speed+=2) {

        double t_err = 0;
        double r_err = 0;
        double num   = 0;

        // for all errors do
        for (KittiError& e:es)
            if (fabs(e.speed-speed)<2.0) {
                t_err += e.t_err;
                r_err += e.r_err;
                ++num;
            }


        // we require at least 3 values
        if (num>2.5) {
            ts<<speed<<" "<<t_err/num<<"\n";
            rs<<speed<<" "<<r_err/num<<"\n";
        }
    }
    //std::string gpdata  =output_path+name+"_"+type+".data";
    std::ofstream fos;
    fos.open(output_path+name+"_tl.data");    fos<<tl.str()<<endl;fos.close();
    fos.open(output_path+name+"_ts.data");    fos<<ts.str()<<endl;fos.close();
    fos.open(output_path+name+"_rl.data");    fos<<rl.str()<<endl;fos.close();
    fos.open(output_path+name+"_rs.data");    fos<<rs.str()<<endl;fos.close();

}

void error_plot(std::string output_path,std::string name,std::string type)
{
    std::string gpfile  =output_path+name+"_"+type+".gp";
    std::string gpdata  =output_path+name+"_"+type+".data";
    std::string gpout   =output_path+name+"_"+type+".eps";
    std::stringstream ss;



    ss<<"set term postscript eps enhanced color\n";
    ss<<"set output \""+gpout+"\"\n";
    ss<<"set size ratio 0.5\n";
    ss<<"set yrange [0:*]\n";
    std::string title,xlabel,odd,ylabel;
    {
        if(type.compare("tl")==0){
            title="Translation Error";
            xlabel="Path Length [m]";
            odd="1:($2*100)";
        }
        if(type.compare("rl")==0){
            title="Rotation Error";
            xlabel="Path Length [m]";
            odd="1:($2*57.3)";
        }
        if(type.compare("ts")==0){
            title="Translation Error";
            xlabel="Speed [km/h]";
            odd="($1*3.6):($2*100)";
        }
        if(type.compare("rs")==0){
            title="Rotation Error";
            xlabel="Speed [km/h]";
            odd="($1*3.6):($2*57.3)";
        }
        ylabel=title;
        if(title.compare("Translation Error")==0)
            ylabel+="  [%%]";
        else
            ylabel+=" [deg/m]";
    }
    ss<<"set xlabel \""<<xlabel<<"\"\n";
    ss<<"set ylabel \""<<ylabel<<"\"\n";

    ss<<"plot \""+gpdata+"\" using "<<odd;
    ss<<" title '"<<title<<"' lc rgb \"#0000FF\" pt 4 w linespoints\n";


    std::ofstream fos;
    fos.open(gpfile);    fos<<ss.str()<<endl;    fos.close();
    std::string command="gnuplot "+gpfile;
    int err=system(command.c_str());assert(!err);if(err) throw new std::logic_error("failed command: "+command);
    command="rm "+gpfile + " "+gpdata;
    err=system(command.c_str());assert(!err);if(err) throw new std::logic_error("failed command: "+command);

}
void plot_errors(std::vector<KittiError>& es, std::string output_path, std::string name,std::vector<double> lengths){

    std::string cmd="mkdir -p "+output_path;
    mlib::saferSystemCall(cmd);

    write_error_data(es,output_path,name,lengths);
    error_plot(output_path,name,"tl");
    error_plot(output_path,name,"rl");
    error_plot(output_path,name,"ts");
    error_plot(output_path,name,"rs");
}







std::vector<cvl::PoseD> invert(std::vector<cvl::PoseD> ps){
    for(PoseD& p:ps)
        p=p.inverse();
    return ps;
}



void plot_sequence(std::vector<cvl::PoseD> gt,std::vector<cvl::PoseD> res,std::string label,
                   std::string output_path,
                   std::string name){
    plot_sequence(gt,{res},{label},output_path,name);
}


void plot_sequence(std::vector<cvl::PoseD> gt,
                   std::vector<std::vector<cvl::PoseD>> posess ,
                   std::vector<std::string> names,
                   std::string output_path,
                   std::string name){


    // check data
    assert(posess.size()>0);
    assert(names.size()==posess.size());
    assert(gt.size()==0 ||gt.size()==posess[0].size());
    for(uint i=0;i<posess.size();++i)
        assert(posess.at(0).size()==posess.at(i).size());


    // invert the poses as they are in kitti format on read...
    gt=invert(gt);
    for(auto& ps:posess)
        ps=invert(ps);

    // compute the min and max vals, for the results and gt
    Vector3d tmin,tmax;
    tmax=tmin=posess.at(0).at(0).getTinW();
    posess.push_back(gt);
    for(auto& ps:posess)
        for(PoseD& p:ps){
            Vector3d t=p.getTinW();
            for(int i=0;i<3;++i){
                tmin(i)=std::min(tmin(i),t(i));
                tmax(i)=std::max(tmax(i),t(i));
            }
        }
    posess.pop_back();// remove gt again...



    // kitti standard is just use the xz plane projection

    // find the edges and move the center of mass to the 00 coordinates, then offset




    // kitti uses a region which is the super of both, but better to just use the right one
    double x_min=tmin(0);
    double y_min=tmin(1);
    double z_min=tmin(2);

    double x_max=tmax(0);
    double y_max=tmax(1);
    double z_max=tmax(2);



// add a 150 or 10% to the borders
    double dfx=std::abs((x_max-x_min)*0.1);    if(dfx<100)        dfx=100;
    x_min-=dfx;    x_max+=dfx;

    double dfy=std::abs((y_max-y_min)*0.1);    if(dfy<100)        dfy=100;
    y_min-=dfy;    y_max+=dfy;

    double dfz=std::abs((z_max-z_min)*0.1);    if(dfz<100)        dfz=100;
    z_min-=dfz;    z_max+=dfz;



    // write the gnuplot data file
    std::string gpfile  =output_path+name+"_path.gp";
    std::string gpdata_gt  =output_path+name+"_path_gt.data";
    std::string gpdata  =output_path+name+"_path.data";

    mlib:: makefilepath(gpfile);
    mlib::makefilepath(gpdata);

    // two files!
    {
        {
            std::stringstream ss;
            for(uint i=0;i<posess[0].size();++i){
                for(uint j=0;j<posess.size();++j){
                    auto t=posess[j][i].getTinW();
                    ss<<t(0) <<" "<<t(1)<<" "<<t(2)<<" ";

                }
                ss<<"\n";
            }
            std::ofstream fos;        fos.open(gpdata);        fos<<ss.str()<<endl;        fos.close();
        }
        if(gt.size()>0)
        {
            std::stringstream ss;
            for(uint i=0;i<gt.size();++i){
                Vector3d t=gt[i].getTinW();
                ss<<t(0) <<" "<<t(1)<<" "<<t(2)<<"\n";
            }
            std::ofstream fos;        fos.open(gpdata_gt);        fos<<ss.str()<<endl;        fos.close();
        }

    }


    std::vector<std::string> colors={"#0000FF","#00FF00","#0000FF"};

    {
        std::string gpout   =output_path+name+"_path.eps";
        // write the gnuplot file x,z
        {
            std::stringstream ss;
            ss<<"set term postscript eps enhanced color\n";
            ss<<"set output \""<<gpout<<"\"\n";
            ss<<"set size ratio -1\n";
            ss<<"set xrange ["<<x_min<<":"<<x_max<<"]\n";
            ss<<"set yrange ["<<z_min<<":"<<z_max<<"]\n";
            ss<<"set xlabel \"x [m]\"\n";
            ss<<"set ylabel \"z [m]\"\n";
            ss<<"plot";
            if(gt.size()>0)
                ss<<"\""<<gpdata_gt<<"\" using 1:3 lc rgb \"#FF0000\" title 'Ground Truth' w lines,";

            for(uint i=0;i<posess.size()&& posess.size()<colors.size();++i)
                ss<<"\""<<gpdata<<"\" using "<<i*3+1<<":"<<i*3+3<<" lc rgb \""<<colors[i]<<"\" title '"<<names[i]<<"' w lines,";



            ss<<"\"< head -1 "<<gpdata<<"\" using 1:3 lc rgb \"#000000\" pt 4 ps 1 lw 2 title 'Sequence Start' w points\n";
            std::ofstream fos;
            fos.open(gpfile);        fos<<ss.str()<<endl;        fos.close();
        }

        // run gnuplot => create png + eps

        std::string command="gnuplot "+ gpfile;
        int err=system(command.c_str());assert(!err);if(err) throw new std::logic_error("failed command: "+command);
        //  command="rm "+gpfile + " "+gpdata;
        //  err=system(command.c_str());assert(!err);if(err) throw new std::logic_error("failed command: "+command);



    }
    {
        std::string gpout   =output_path+name+"_xy_path.eps";
        // write the gnuplot file x,y
        {
            std::stringstream ss;
            ss<<"set term postscript eps enhanced color\n";
            ss<<"set output \""<<gpout<<"\"\n";
            ss<<"set size ratio -1\n";
            ss<<"set xrange ["<<x_min<<":"<<x_max<<"]\n";
            ss<<"set yrange ["<<y_min<<":"<<y_max<<"]\n";
            ss<<"set xlabel \"x [m]\"\n";
            ss<<"set ylabel \"y [m]\"\n";
            ss<<"plot";
            if(gt.size()>0)
                ss<<"\""<<gpdata_gt<<"\" using 1:2 lc rgb \"#FF0000\" title 'Ground Truth' w lines,";

            for(uint i=0;i<posess.size()&& posess.size()<colors.size();++i)
                ss<<"\""<<gpdata<<"\" using "<<i*3+1<<":"<<i*3+2<<" lc rgb \""<<colors[i]<<"\" title '"<<names[i]<<"' w lines,";



            ss<<"\"< head -1 "<<gpdata<<"\" using 1:2 lc rgb \"#000000\" pt 4 ps 1 lw 2 title 'Sequence Start' w points\n";
            std::ofstream fos;
            fos.open(gpfile);        fos<<ss.str()<<endl;        fos.close();
        }

        // run gnuplot => create png + eps

        std::string command="gnuplot "+ gpfile;
        int err=system(command.c_str());assert(!err);if(err) throw new std::logic_error("failed command: "+command);
        //  command="rm "+gpfile + " "+gpdata;
        //  err=system(command.c_str());assert(!err);if(err) throw new std::logic_error("failed command: "+command);



    }




    //cout<<"Plot - Done"<<endl;
}



















}// end kitti namespace
}// end namespace cvl
