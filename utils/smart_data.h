#pragma once
/* ********************************* FILE ************************************/
/** \file    smart_data.hpp
 *
 * \brief    RAII new basically...
 * \todo
 *  - fix cuda support using templated delete
 *
 * \remark
 * - c++11
 * - deprecated
 *
 *
 *        c++11 and onwards:
 *      std::shared_ptr<int> sp(new int[10], [](int const *p) { delete[] p;  });
 *      std::shared_ptr<int> sp(new int[10], [](int const *p) { cuda_free (p);});
 *
 *      problem with if I dont know if it was allocated with new or new [] is unresolvable...
 *      smart_ptr<T>(T * t, [](int const *p) { delete[] p;  }) can wrap anything, but you have to know if its [] or not
 *
 *
 ******************************************************************************/



