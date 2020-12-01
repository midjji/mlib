#pragma once
#include <mlib/utils/cvl/matrix_adapter.h>
#include <vector>

namespace cvl{
/**
 * @brief The MemManager class
 * convenience class for memory management...
 * allocate takes care of stride alignment automatically, guaranteeing a 256 byte alignment
 * but returned matrixes likely to be continuous
 * dont use for small matrixes
 *
 * \todo
 * - figure out why this can cause problems in combination with vector...
 * - make static allocate classes
 * - generalize for ND matrixes?
 *
 */
class MemManager{
public:
    MemManager(){allocs.reserve(2048);}
    ~MemManager(){

        for(uint i=0;i<allocs.size();++i){
            delete allocs[i];
        }
    }


    //template<class T> static MatrixAdapter<T>



    template<class T>
    /**
     * @brief allocate
     * @param rows
     * @param cols
     * @return a automatically strided matrix! stride size is set to 256 which might be a bit much on cpu but you really shouldnt use this matrix class for small matrixes anyways
     */
    MatrixAdapter<T> allocate(uint rows, uint cols){
        assert(rows>0);
        assert(cols>0);

        auto m=MatrixAdapter<T>::allocate(rows,cols);
        manage(m);
        return m;
    }


    template<class T> void manage(MatrixAdapter<T>& m){allocs.push_back((unsigned char*)m.getData());}
    template<class T> void manage(T* data){allocs.push_back((unsigned char*)data);}





private:
    std::vector<unsigned char*> allocs;

};

}// end namespace cvl
