#include "sys.h"
#include <sstream>


namespace mlib{


/**
 * @brief saferSystemCall
 * @param cmd
 * \todo
 * - improve if you figure out how to remove dangerous commands...
 */
int saferSystemCall(std::string cmd){
//#warning "System calls are notoriously horrible, do you really need to use one?!"
    int err=system(cmd.c_str());
    if(err!=0){
        std::stringstream ss;
        ss << "Failed system call:\n Error: "<<err<<"\n "<<cmd<<"\n";
          //throw new std::runtime_error(ss.str());
    }
    return err;
}

}// end namespace mlib
