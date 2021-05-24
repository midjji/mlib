#include <map>
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>


#include "jkqtplotter/jkqtplotter.h"
#include "jkqtplotter/graphs/jkqtpscatter.h"
#include <mlib/utils/mlog/log.h>
#include <mtgui.h>

namespace cvl {


namespace  {
class Plotter{
public:
    void clear_plot(std::string title){
        auto plotter=self.lock();
        run_in_gui_thread(
                    new QAppLambda([plotter,title](){
            plotter->clear_plot_internal(title);}));
    }
    void plot(const std::vector<double>& xs,
              const std::vector<double>& ys,
              std::string title, std::string label){
        call_plot(xs,ys,title, label);
    }

    void plot(const std::vector<double>& ts, const std::map<std::string, std::vector<double>>& errs, std::string title){
        for(auto& [name,err]:errs) call_plot(ts,err,title, name);
    }
    static std::shared_ptr<Plotter> create() {
        auto plotter= std::shared_ptr<Plotter>(new Plotter);
        plotter->self=plotter;
        return plotter;
    }

private:
    Plotter() {}

    std::weak_ptr<Plotter> self;

    std::mutex call_plot_mutex;
    void call_plot(std::vector<double> xs,
                   std::vector<double> ys,
                   std::string title,
                   std::string label)
    {

        std::unique_lock<std::mutex> ul(call_plot_mutex);
        // Here I know the plotter exists for as long as the
        // capture this by reference, then run it in blocking mode, so the lambda finishes before the function returns
        // actually opencv does not like blocking! hmm???
        auto plotter=self.lock();
        run_in_gui_thread(
                    new QAppLambda([plotter,xs,ys,title,label](){
            plotter->plot_internal(xs,ys,title, label);}));
    }


    std::map<std::string, std::shared_ptr<JKQTPlotter>> plots_;
    std::shared_ptr<JKQTPlotter> named_plot(std::string name){
        auto it=plots_.find(name);
        if(it!=plots_.end()) return it->second;
        auto plot=std::make_shared<JKQTPlotter>();
        plots_[name]=plot;
        plot->setWindowTitle(QString(name.c_str()));
        // show plotter and make it a decent size
        plot->show();
        plot->resize(600,400);
        return plot;
    }
    void clear_plot_internal(std::string title){
        std::unique_lock<std::mutex> ul(plot_internal_mtx);
        auto it=plots_.find(title);
        if(it!=plots_.end()) {
            //plots_.erase(title);// slow
            it->second->clearGraphs();
        }
    }

    std::mutex plot_internal_mtx; // protects the map
    void plot_internal(
            const std::vector<double>& xs,
            const std::vector<double>& ys,
            std::string title, // for the window
            std::string label // for this graph
            ){
        // must be run in the event loop, so call via call_plot
        std::unique_lock<std::mutex> ul(plot_internal_mtx);


        // 1. create a plotter window and get a pointer to the internal datastore (for convenience)
        std::shared_ptr<JKQTPlotter> plot=named_plot(title);
        JKQTPDatastore* ds=plot->getDatastore();

        // 2. now we create data for a simple plot (a sine curve)
        QVector<double> X, Y;
        X.reserve(xs.size());
        Y.reserve(ys.size());
        bool nansinplot=false;
        for (uint i=0;i<std::min(xs.size(),ys.size());++i) {
            if(std::isnan(xs[i]+ys[i])){nansinplot=true; continue;}
            X<<xs[i];
            Y<<ys[i];
        }
        if(nansinplot)
            std::cout<<"nans in plot"<<std::endl;

        // 3. make data available to JKQTPlotter by adding it to the internal datastore.
        //    Note: In this step the data is copied (of not specified otherwise), so you can
        //          reuse X and Y afterwards!
        //    the variables columnX and columnY will contain the internal column ID of the newly
        //    created columns with names "x" and "y" and the (copied) data from X and Y.
        ds->clear();
        size_t columnX=ds->addCopiedColumn(X, "x");
        size_t columnY=ds->addCopiedColumn(Y, "y");

        // 4. create a graph in the plot, which plots the dataset X/Y:
        JKQTPXYLineGraph* graph1=new JKQTPXYLineGraph(plot.get());
        graph1->setXColumn(columnX);
        graph1->setYColumn(columnY);
        graph1->setTitle(QObject::tr(label.c_str()));


        // 5. add the graph to the plot, so it is actually displayed
        plot->addGraph(graph1);
        // 6. autoscale the plot so the graph is contained
        plot->zoomToFit();

    }
};

std::mutex plotter_mtx;
std::shared_ptr<Plotter> plotter_=nullptr;
}
std::shared_ptr<Plotter> plotter(){
    std::unique_lock<std::mutex> ul(plotter_mtx);
    if(plotter_==nullptr)
        plotter_=Plotter::create();
    return plotter_;
}

void clear_plot(std::string title){
    plotter()->clear_plot(title);
}
void plot(const std::vector<double>& ys,
          std::string title,
          std::string label){
    std::vector<double> indexes;indexes.reserve(ys.size());
    for(uint i=0;i<ys.size();++i)
        indexes.push_back(i);
    plotter()->plot(indexes,ys,title, label);
}
void plot(const std::vector<double>& xs,
          const std::vector<double>& ys,
          std::string title,
          std::string label){
    plotter()->plot(xs,ys,title, label);
}
void plot(
        const std::vector<double>& xs,
        const std::map<std::string, std::vector<double>>& yss,
        std::string title)
{
    if(yss.size()==0)
        std::cout<<"missing y values for plot!"<<std::endl;
    plotter()->plot(xs,yss, title);
}
void initialize_plotter(){
    plotter();
}
}





