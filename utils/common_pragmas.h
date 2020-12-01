#pragma once

#if defined(__clang__)
    /* Clang/LLVM. ---------------------------------------------- */
#elif defined(__ICC) || defined(__INTEL_COMPILER)
    /* Intel ICC/ICPC. ------------------------------------------ */
#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    /* GNU GCC/G++. --------------------------------------------- */
#elif defined(_MSC_VER)
    /* Microsoft Visual Studio. --------------------------------- */
#endif

