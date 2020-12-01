#pragma once
//////////////////////////////////////////////////////////////////////////
//		Constants
/// \todo replace with numeric_limits, #include <limits>
#include <limits>





namespace cvl{

    const double eps = 2.220446049250313E-016;
    const double epsDouble = 2.220446049250313E-016;

    // to avoid numerical errors for i.e. divisions
    const double zeroToleranceDouble = 1e-08;

//	const double infDouble = HUGE_VAL;

    //const float infFloat = float(infDouble);	;

    const int infInt = 2147483647;		// hmm ?? only 32 bits long in windows

    const int infLong = 2147483647;		// hmm ?? only 32 bits long in windows

    const long double pi = 3.141592653589793238462643383279502884L;


    // const double nan = sqrt(-1.0);
    const unsigned long nanRaw[2]={0xffffffff, 0x7fffffff};
//	const double nan = *( double* )nanRaw;

    // math.h pi in c++ and c is just 5 digits, seems dumb
    const float pi_f = float(pi); // default M_PI is just 5 digits?
    const double pi_d = double(pi); // default M_PI is just 5 digits?

}


typedef unsigned int uint;
