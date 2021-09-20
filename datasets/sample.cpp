#include <mlib/datasets/sample.h>
#include <mlib/datasets/stereo_sequence.h>

static_assert (sizeof(float128) ==16, "long float128 implementation is missing" );
namespace cvl{
Sample::Sample(float128 time, const StereoSequence* ss):
    time_(time),wseq(ss){}
Sample::~Sample(){}

float128 Sample::time() const{return time_;}
const StereoSequence* Sample::sequence() const{
    return wseq;
}

}
