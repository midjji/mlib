#pragma once
#include <limits>
namespace cvl {


template<class Above>
/**
 * @brief binary_search:  guided search for the highest value which satisfies !above
 * @param min: must be in range and !above
 * @param max: must be in range and above
 * @param above
 * @param current
 * @return highest !above (supremum)
 *
 * int res=binary_search(-5533, 11000001,[](int v){return v<500;},2000);
 */
int binary_search(Above above,
                  int min=std::numeric_limits<int>::min(),
                  int max=std::numeric_limits<int>::max(),
                  int current=0, int max_steps=65){
    if(current==max ||current ==min) return current;
    if(above(current))
        min=current;
    else
        max=current;

    // recursion is fine, it wont ever take longer than 64 tries
    // unless above is broken
    if(max_steps<0) return 0; // hang on this for broken above, or warn?
    return binary_search(above,min, max,(max-min)/2 + min,max_steps--);
}


}
