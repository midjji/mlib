#include <mlib/datasets/sample.h>
#include <mlib/datasets/stereo_sequence.h>

static_assert (sizeof(float128) ==16, "long float128 implementation is missing" );
namespace cvl{
Sample::Sample(float128 time, std::shared_ptr<StereoSequence> ss):
    time_(time),wseq(ss){}
Sample::~Sample(){}

float128 Sample::time() const{return time_;}
const std::shared_ptr<StereoSequence>Sample::sequence() const{
    return wseq;
}

}
