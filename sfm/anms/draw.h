#pragma once
#if 0
This version does not keep the locked ones,
Though they differ a bit in performance,
keeping more than one implementation is uneccessary at this time,
Also the interface needs to be redesigned!


#include <memory>
#include <mlib/sfm/anms/base.h>
#include <mlib/utils/memmanager.h>
namespace cvl{
namespace anms{

class DrawSolver : public Solver{
public:
    DrawSolver(int rows, int cols, double maxRadius);
    ~DrawSolver();
    void init(const std::vector<Data>& datas,const std::vector<Data>& locked) override;


    void compute (double minRadius, int minKeep) override ;
    bool exact() override;
    void showMask();

    int rows,cols;
    int min2keep=450;
    double maxRadius;
private:
    std::vector<std::uint8_t> data;
};





} // end namespace anms
}// end namespace cvl

#endif
