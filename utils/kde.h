#pragma once
#include <vector>
namespace cvl{

struct KDE{
    // provides the pdf, which integrates to 1,
    // not probability!
    std::vector<double> datas, datas_sorted;
    double low_bound(double ratio=0.01) const;
    double high_bound(double ratio=0.99) const;
    std::tuple<std::vector<double>, std::vector<double>>
    plot(int N=100, double sigma=1) const;
    std::tuple<std::vector<double>, std::vector<double>>
    plot(double low,
         double high,
         int N=100, double sigma=1) const;
    KDE(const std::vector<double>& datas);
    double operator()(double t, double sigma=1) const;
};

struct Histogram{
    std::vector<double> datas;
    Histogram(const std::vector<double>& data);
    std::tuple<std::vector<double>, std::vector<double>>
    plot(double low,
         double high,
         int N=100,
         int buckets=100) const;
};

}
