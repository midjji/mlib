#include <mlib/utils/spline/coeffs.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/utils/spline/coeffs_extra.h>
namespace cvl {
double cardinal_basis(double x, int degree, int derivative)
{
    if(x<0.0) return 0.0;

    // derivative ==0, the common case...
    if(derivative==0)
    {
        if(x<0.0)
            return 0.0;

        // from 0 to 1 is 1
        if(degree==0){
            if(x<1.0) return 1.0;
            return 0.0;
        }

        if(degree==1)
        {
            if(x<1.0) return x;
            if(x<2.0) return 2.0-x;
            return 0.0;
        }
        if(degree==2){
            if(x<1.0) return 0.5*x*x;
            if(x<2.0) return 0.5*(-2.0*x*x + 6.0*x -3.0);
            if(x<3.0) return 0.5*(x - 3.0)*(x - 3.0);
            return 0.0;
        }
        if(degree==3){
            if(x<1.0) return (1.0/6.0)*x*x*x;
            if(x<2.0) return (1.0/6.0)*(-3.0*x*x*x +12.0*x*x - 12.0*x+4.0);
            if(x<3.0) return (1.0/6.0)*(3.0*x*x*x -24.0*x*x +60*x -44.0);
            if(x<4.0) return (1.0/6.0)*(-x +4.0)*(-x +4.0)*(-x +4.0);
            return 0.0;
        }
        if(degree==4){
            if(x<1.0) return (1.0/24.0)*x*x*x*x;
            if(x<2.0) return (1.0/24.0)*(-4.0*x*x*x*x +20.0*x*x*x -30.0*x*x +20.0*x -5.0);
            if(x<3.0) return (1.0/24.0)*(6.0*x*x*x*x  -60.0*x*x*x +210.0*x*x -300.0*x +155.0);
            if(x<4.0) return (1.0/24.0)*(-4.0*x*x*x*x + 60.0*x*x*x -330.0*x*x +780.0*x - 655.0);
            if(x<5.0) return (1.0/24.0)*(x - 5.0)*(x - 5.0)*(x - 5.0)*(x - 5.0);
            return 0.0;
        }
    }
//mlog()<<x<<" "<<degree<< " "<<derivative<<" \n";
    //assert(degree>=0);
    if(degree<0)    {
        mlog()<<"gmm=?"<<degree<< " "<<derivative<<"\n"; exit(1);

   }
    // special case, degree<0, derivative==0
    if(degree<0 && derivative==0)
    {
        //return 0;

        // a sortof correct description, it will only work if the user really knows what they are doing...
        if(x==0) return 1;

        if(x==1) return -1;// no this part is wrong, but I kinda need the delta, hmm should never use this
        return 0;
    }

    if(derivative>0)
    {
        return cardinal_basis(x,degree-1, derivative-1)
                - cardinal_basis(x-1.0,degree-1,derivative-1);
    }
    if(derivative<0){
        // assume the user is also summing over all values!
        // beware!
        if(derivative==-1) return cumulative_cardinal_basis(x,degree,0);
        //if(derivative==-2) return forward_extrapolation_cumulative_cardinal_basis(x,degree,0);
        mlog()<<"ill supported"<<derivative<<" "<<degree<< " "<<x << "\n";
    }




    return (cardinal_basis(x,degree-1,derivative)*x + cardinal_basis(x-1, degree-1,derivative)*(degree+1.0-x))/double(degree);
}

double cardinal_basis(double x, double y, int degree, int derivative){
    //Not multiplied, convolved;
    return 0;
    return cardinal_basis(x,degree, derivative)*cardinal_basis(y,degree,derivative);
}

double cumulative_cardinal_basis(double x, int degree, int derivative){


   //Note, degree above 10 has noticeable numerical errors,
    // a fundamentally different approach would be required.







    if(x<0) return 0;
    if(derivative>0)
    {
        return cardinal_basis(x,degree-1,derivative-1);
    }

    if(degree>10){
        mlog()<<"unsupported! "<<degree<<"\n";
        exit(1);
    }
    if(degree<0){
        mlog()<<"unsupported! "<<degree<<"\n";
        exit(1);
    }
    std::array<double, 20> xps; // wont work numerically for above 20 anyways...
    xps[0] = x;
    for(int i=1;i<=degree && i <20;++i)
        xps[i] =xps[i-1]*x;

    auto xp=[&](int i)
    {
        return xps[i-1];
    };
    ////////////////////////////
    if(degree==0){/*
    [0, inf)p(x) = 1
    */
        double v=0;
        if(x>=0){
            double val=1;
            v+=val;
        }

        return v;
    }

    ////////////////////////////
    if(degree==1){/*
    [0, 1)p(x) = 1x
    [1, inf)p(x) = 1
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(1) ;
            v+=val;
        }

        if(1<=x){
            double val=1;
            v+=val;
        }

        return v;
    }

    ////////////////////////////
    if(degree==2){/*
    [0, 1)p(x) = (1x^2)/2
    [1, 2)p(x) = (-1x^2 + 4x + -2)/2
    [2, inf)p(x) = (2)/2
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(2) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=-2+ 4*xp(1) - xp(2) ;
            v+=val;
        }

        if(2<=x){
            double val=2;
            v+=val;
        }

        return v/double(2);
    }

    ////////////////////////////
    if(degree==3){/*
    [0, 1)p(x) = (1x^3)/6
    [1, 2)p(x) = (-2x^3 + 9x^2 + -9x + 3)/6
    [2, 3)p(x) = (1x^3 + -9x^2 + 27x + -21)/6
    [3, inf)p(x) = (6)/6
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(3) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=3- 9*xp(1) + 9*xp(2) - 2*xp(3) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=-21+ 27*xp(1) - 9*xp(2) + xp(3) ;
            v+=val;
        }

        if(3<=x){
            double val=6;
            v+=val;
        }

        return v/double(6);
    }

    ////////////////////////////
    if(degree==4){/*
    [0, 1)p(x) = (1x^4)/24
    [1, 2)p(x) = (-3x^4 + 16x^3 + -24x^2 + 16x + -4)/24
    [2, 3)p(x) = (3x^4 + -32x^3 + 120x^2 + -176x + 92)/24
    [3, 4)p(x) = (-1x^4 + 16x^3 + -96x^2 + 256x + -232)/24
    [4, inf)p(x) = (24)/24
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(4) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=-4+ 16*xp(1) - 24*xp(2) + 16*xp(3) - 3*xp(4) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=92- 176*xp(1) + 120*xp(2) - 32*xp(3) + 3*xp(4) ;
            v+=val;
        }

        if(3<=x && x<4){
            double val=-232+ 256*xp(1) - 96*xp(2) + 16*xp(3) - 1*xp(4) ;
            v+=val;
        }

        if(4<=x){
            double val=24;
            v+=val;
        }

        return v/double(24);
    }

    ////////////////////////////
    if(degree==5){/*
    [0, 1)p(x) = (1x^5)/120
    [1, 2)p(x) = (-4x^5 + 25x^4 + -50x^3 + 50x^2 + -25x + 5)/120
    [2, 3)p(x) = (6x^5 + -75x^4 + 350x^3 + -750x^2 + 775x + -315)/120
    [3, 4)p(x) = (-4x^5 + 75x^4 + -550x^3 + 1950x^2 + -3275x + 2115)/120
    [4, 5)p(x) = (1x^5 + -25x^4 + 250x^3 + -1250x^2 + 3125x + -3005)/120
    [5, inf)p(x) = (120)/120
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(5) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=5- 25*xp(1) + 50*xp(2) - 50*xp(3) + 25*xp(4) - 4*xp(5) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=-315+ 775*xp(1) - 750*xp(2) + 350*xp(3) - 75*xp(4) + 6*xp(5) ;
            v+=val;
        }

        if(3<=x && x<4){
            double val=2115- 3275*xp(1) + 1950*xp(2) - 550*xp(3) + 75*xp(4) - 4*xp(5) ;
            v+=val;
        }

        if(4<=x && x<5){
            double val=-3005+ 3125*xp(1) - 1250*xp(2) + 250*xp(3) - 25*xp(4) + xp(5) ;
            v+=val;
        }

        if(5<=x){
            double val=120;
            v+=val;
        }

        return v/double(120);
    }

    ////////////////////////////
    if(degree==6){/*
    [0, 1)p(x) = (1x^6)/720
    [1, 2)p(x) = (-5x^6 + 36x^5 + -90x^4 + 120x^3 + -90x^2 + 36x + -6)/720
    [2, 3)p(x) = (10x^6 + -144x^5 + 810x^4 + -2280x^3 + 3510x^2 + -2844x + 954)/720
    [3, 4)p(x) = (-10x^6 + 216x^5 + -1890x^4 + 8520x^3 + -20790x^2 + 26316x + -13626)/720
    [4, 5)p(x) = (5x^6 + -144x^5 + 1710x^4 + -10680x^3 + 36810x^2 + -65844x + 47814)/720
    [5, 6)p(x) = (-1x^6 + 36x^5 + -540x^4 + 4320x^3 + -19440x^2 + 46656x + -45936)/720
    [6, inf)p(x) = (720)/720
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(6) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=-6+ 36*xp(1) - 90*xp(2) + 120*xp(3) - 90*xp(4) + 36*xp(5) - 5*xp(6) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=954- 2844*xp(1) + 3510*xp(2) - 2280*xp(3) + 810*xp(4) - 144*xp(5) + 10*xp(6) ;
            v+=val;
        }

        if(3<=x && x<4){
            double val=-13626+ 26316*xp(1) - 20790*xp(2) + 8520*xp(3) - 1890*xp(4) + 216*xp(5) - 10*xp(6) ;
            v+=val;
        }

        if(4<=x && x<5){
            double val=47814- 65844*xp(1) + 36810*xp(2) - 10680*xp(3) + 1710*xp(4) - 144*xp(5) + 5*xp(6) ;
            v+=val;
        }

        if(5<=x && x<6){
            double val=-45936+ 46656*xp(1) - 19440*xp(2) + 4320*xp(3) - 540*xp(4) + 36*xp(5) - 1*xp(6) ;
            v+=val;
        }

        if(6<=x){
            double val=720;
            v+=val;
        }

        return v/double(720);
    }

    ////////////////////////////
    if(degree==7){/*
    [0, 1)p(x) = (1x^7)/5040
    [1, 2)p(x) = (-6x^7 + 49x^6 + -147x^5 + 245x^4 + -245x^3 + 147x^2 + -49x + 7)/5040
    [2, 3)p(x) = (15x^7 + -245x^6 + 1617x^5 + -5635x^4 + 11515x^3 + -13965x^2 + 9359x + -2681)/5040
    [3, 4)p(x) = (-20x^7 + 490x^6 + -4998x^5 + 27440x^4 + -87710x^3 + 164640x^2 + -169246x + 73864)/5040
    [4, 5)p(x) = (15x^7 + -490x^6 + 6762x^5 + -50960x^4 + 225890x^3 + -588000x^2 + 834274x + -499576)/5040
    [5, 6)p(x) = (-6x^7 + 245x^6 + -4263x^5 + 40915x^4 + -233485x^3 + 790125x^2 + -1.4626e+06x + 1.14105e+06)/5040
    [6, 7)p(x) = (1x^7 + -49x^6 + 1029x^5 + -12005x^4 + 84035x^3 + -352947x^2 + 823543x + -818503)/5040
    [7, inf)p(x) = (5040)/5040
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ 1*xp(7) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=7- 49*xp(1) + 147*xp(2) - 245*xp(3) + 245*xp(4) - 147*xp(5) + 49*xp(6) - 6*xp(7) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=-2681+ 9359*xp(1) - 13965*xp(2) + 11515*xp(3) - 5635*xp(4) + 1617*xp(5) - 245*xp(6) + 15*xp(7) ;
            v+=val;
        }

        if(3<=x && x<4){
            double val=73864- 169246*xp(1) + 164640*xp(2) - 87710*xp(3) + 27440*xp(4) - 4998*xp(5) + 490*xp(6) - 20*xp(7) ;
            v+=val;
        }

        if(4<=x && x<5){
            double val=-499576+ 834274*xp(1) - 588000*xp(2) + 225890*xp(3) - 50960*xp(4) + 6762*xp(5) - 490*xp(6) + 15*xp(7) ;
            v+=val;
        }

        if(5<=x && x<6){
            double val=1141049- 1462601*xp(1) + 790125*xp(2) - 233485*xp(3) + 40915*xp(4) - 4263*xp(5) + 245*xp(6) - 6*xp(7) ;
            v+=val;
        }

        if(6<=x && x<7){
            double val=-818503+ 823543*xp(1) - 352947*xp(2) + 84035*xp(3) - 12005*xp(4) + 1029*xp(5) - 49*xp(6) + 1*xp(7) ;
            v+=val;
        }

        if(7<=x){
            double val=5040;
            v+=val;
        }

        return v/double(5040);
    }

    ////////////////////////////
    if(degree==8){/*
    [0, 1)p(x) = (1x^8)/40320
    [1, 2)p(x) = (-7x^8 + 64x^7 + -224x^6 + 448x^5 + -560x^4 + 448x^3 + -224x^2 + 64x + -8)/40320
    [2, 3)p(x) = (21x^8 + -384x^7 + 2912x^6 + -12096x^5 + 30800x^4 + -49728x^3 + 49952x^2 + -28608x + 7160)/40320
    [3, 4)p(x) = (-35x^8 + 960x^7 + -11200x^6 + 72576x^5 + -286720x^4 + 712320x^3 + -1.09312e+06x^2 + 951168x + -360256)/40320
    [4, 5)p(x) = (35x^8 + -1280x^7 + 20160x^6 + -178304x^5 + 967680x^4 + -3.30176e+06x^3 + 6.93504e+06x^2 + -8.22387e+06x + 4.22726e+06)/40320
    [5, 6)p(x) = (-21x^8 + 960x^7 + -19040x^6 + 213696x^5 + -1.48232e+06x^4 + 6.49824e+06x^3 + -1.7565e+07x^2 + 2.67761e+07x + -1.76477e+07)/40320
    [6, 7)p(x) = (7x^8 + -384x^7 + 9184x^6 + -124992x^5 + 1.05784e+06x^4 + -5.69453e+06x^3 + 1.90133e+07x^2 + -3.59295e+07x + 2.93815e+07)/40320
    [7, 8)p(x) = (-1x^8 + 64x^7 + -1792x^6 + 28672x^5 + -286720x^4 + 1.83501e+06x^3 + -7.34003e+06x^2 + 1.67772e+07x + -1.67369e+07)/40320
    [8, inf)p(x) = (40320)/40320
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ 1*xp(8) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=-8+ 64*xp(1) - 224*xp(2) + 448*xp(3) - 560*xp(4) + 448*xp(5) - 224*xp(6) + 64*xp(7) - 7*xp(8) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=7160- 28608*xp(1) + 49952*xp(2) - 49728*xp(3) + 30800*xp(4) - 12096*xp(5) + 2912*xp(6) - 384*xp(7) + 21*xp(8) ;
            v+=val;
        }

        if(3<=x && x<4){
            double val=-360256+ 951168*xp(1) - 1093120*xp(2) + 712320*xp(3) - 286720*xp(4) + 72576*xp(5) - 11200*xp(6) + 960*xp(7) - 35*xp(8) ;
            v+=val;
        }

        if(4<=x && x<5){
            double val=4227264- 8223872*xp(1) + 6935040*xp(2) - 3301760*xp(3) + 967680*xp(4) - 178304*xp(5) + 20160*xp(6) - 1280*xp(7) + 35*xp(8) ;
            v+=val;
        }

        if(5<=x && x<6){
            double val=-17647736+ 26776128*xp(1) - 17564960*xp(2) + 6498240*xp(3) - 1482320*xp(4) + 213696*xp(5) - 19040*xp(6) + 960*xp(7) - 21*xp(8) ;
            v+=val;
        }

        if(6<=x && x<7){
            double val=29381512- 35929536*xp(1) + 19013344*xp(2) - 5694528*xp(3) + 1057840*xp(4) - 124992*xp(5) + 9184*xp(6) - 384*xp(7) + 7*xp(8) ;
            v+=val;
        }

        if(7<=x && x<8){
            double val=-16736896+ 16777216*xp(1) - 7340032*xp(2) + 1835008*xp(3) - 286720*xp(4) + 28672*xp(5) - 1792*xp(6) + 64*xp(7) - 1*xp(8) ;
            v+=val;
        }

        if(8<=x){
            double val=40320;
            v+=val;
        }

        return v/double(40320);
    }

    ////////////////////////////
    if(degree==9){/*
    [0, 1)p(x) = (1x^9)/362880
    [1, 2)p(x) = (-8x^9 + 81x^8 + -324x^7 + 756x^6 + -1134x^5 + 1134x^4 + -756x^3 + 324x^2 + -81x + 9)/362880
    [2, 3)p(x) = (28x^9 + -567x^8 + 4860x^7 + -23436x^6 + 71442x^5 + -144018x^4 + 192780x^3 + -165564x^2 + 82863x + -18423)/362880
    [3, 4)p(x) = (-56x^9 + 1701x^8 + -22356x^7 + 167076x^6 + -785862x^5 + 2.42789e+06x^4 + -4.95104e+06x^3 + 6.44792e+06x^2 + -4.87725e+06x + 1.63495e+06)/362880
    [4, 5)p(x) = (70x^9 + -2835x^8 + 50220x^7 + -510300x^6 + 3.27839e+06x^5 + -1.38291e+07x^4 + 3.8401e+07x^3 + -6.78699e+07x^2 + 6.94406e+07x + -3.13952e+07)/362880
    [5, 6)p(x) = (-56x^9 + 2835x^8 + -63180x^7 + 812700x^6 + -6.64411e+06x^5 + 3.57834e+07x^4 + -1.26974e+08x^3 + 2.86505e+08x^2 + -3.73528e+08x + 2.14699e+08)/362880
    [6, 7)p(x) = (28x^9 + -1701x^8 + 45684x^7 + -711396x^6 + 7.07276e+06x^5 + -4.65178e+07x^4 + 2.02231e+08x^3 + -5.60021e+08x^2 + 8.96262e+08x + -6.31828e+08)/362880
    [7, 8)p(x) = (-8x^9 + 567x^8 + -17820x^7 + 325836x^6 + -3.81818e+06x^5 + 2.97187e+07x^4 + -1.5354e+08x^3 + 5.0729e+08x^2 + -9.71534e+08x + 8.20902e+08)/362880
    [8, 9)p(x) = (1x^9 + -81x^8 + 2916x^7 + -61236x^6 + 826686x^5 + -7.44017e+06x^4 + 4.4641e+07x^3 + -1.72187e+08x^2 + 3.8742e+08x + -3.87058e+08)/362880
    [9, inf)p(x) = (362880)/362880
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(9) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=9- 81*xp(1) + 324*xp(2) - 756*xp(3) + 1134*xp(4) - 1134*xp(5) + 756*xp(6) - 324*xp(7) + 81*xp(8) - 8*xp(9) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=-18423+ 82863*xp(1) - 165564*xp(2) + 192780*xp(3) - 144018*xp(4) + 71442*xp(5) - 23436*xp(6) + 4860*xp(7) - 567*xp(8) + 28*xp(9) ;
            v+=val;
        }

        if(3<=x && x<4){
            double val=1634949- 4877253*xp(1) + 6447924*xp(2) - 4951044*xp(3) + 2427894*xp(4) - 785862*xp(5) + 167076*xp(6) - 22356*xp(7) + 1701*xp(8) - 56*xp(9) ;
            v+=val;
        }

        if(4<=x && x<5){
            double val=-31395195+ 69440571*xp(1) - 67869900*xp(2) + 38401020*xp(3) - 13829130*xp(4) + 3278394*xp(5) - 510300*xp(6) + 50220*xp(7) - 2835*xp(8) + 70*xp(9) ;
            v+=val;
        }

        if(5<=x && x<6){
            double val=214698555- 373528179*xp(1) + 286505100*xp(2) - 126973980*xp(3) + 35783370*xp(4) - 6644106*xp(5) + 812700*xp(6) - 63180*xp(7) + 2835*xp(8) - 56*xp(9) ;
            v+=val;
        }

        if(6<=x && x<7){
            double val=-631827909+ 896261517*xp(1) - 560021364*xp(2) + 202230756*xp(3) - 46517814*xp(4) + 7072758*xp(5) - 711396*xp(6) + 45684*xp(7) - 1701*xp(8) + 28*xp(9) ;
            v+=val;
        }

        if(7<=x && x<8){
            double val=820901943- 971534007*xp(1) + 507290364*xp(2) - 153539820*xp(3) + 29718738*xp(4) - 3818178*xp(5) + 325836*xp(6) - 17820*xp(7) + 567*xp(8) - 8*xp(9) ;
            v+=val;
        }

        if(8<=x && x<9){
            double val=-387057609+ 387420489*xp(1) - 172186884*xp(2) + 44641044*xp(3) - 7440174.0000000001*xp(4) + 826686.00000000001*xp(5) - 61236*xp(6) + 2916*xp(7) - 80.999999999999999*xp(8) + xp(9) ;
            v+=val;
        }

        if(9<=x){
            double val=362880;
            v+=val;
        }

        return v/double(362880);
    }

    ////////////////////////////
    if(degree==10){/*
    [0, 1)p(x) = (1x^10)/3.6288e+06
    [1, 2)p(x) = (-9x^10 + 100x^9 + -450x^8 + 1200x^7 + -2100x^6 + 2520x^5 + -2100x^4 + 1200x^3 + -450x^2 + 100x + -10)/3.6288e+06
    [2, 3)p(x) = (36x^10 + -800x^9 + 7650x^8 + -42000x^7 + 149100x^6 + -360360x^5 + 602700x^4 + -690000x^3 + 517950x^2 + -230300x + 46070)/3.6288e+06
    [3, 4)p(x) = (-84x^10 + 2800x^9 + -40950x^8 + 346800x^7 + -1.8921e+06x^6 + 6.98796e+06x^5 + -1.77681e+07x^4 + 3.08028e+07x^3 + -3.49115e+07x^2 + 2.33893e+07x + -7.03981e+06)/3.6288e+06
    [4, 5)p(x) = (126x^10 + -5600x^9 + 110250x^8 + -1.266e+06x^7 + 9.3975e+06x^6 + -4.72021e+07x^5 + 1.62866e+08x^4 + -3.82074e+08x^3 + 5.84404e+08x^2 + -5.27113e+08x + 2.13161e+08)/3.6288e+06
    [5, 6)p(x) = (-126x^10 + 7000x^9 + -173250x^8 + 2.514e+06x^7 + -2.36775e+07x^6 + 1.51248e+08x^5 + -6.64009e+08x^4 + 1.98043e+09x^3 + -3.84528e+09x^2 + 4.39476e+09x + -2.24778e+09)/3.6288e+06
    [6, 7)p(x) = (84x^10 + -5600x^9 + 166950x^8 + -2.9292e+06x^7 + 3.34761e+07x^6 + -2.60258e+08x^5 + 1.39352e+09x^4 + -5.07396e+09x^3 + 1.20271e+10x^2 + -1.67684e+10x + 1.04501e+10)/3.6288e+06
    [7, 8)p(x) = (-36x^10 + 2800x^9 + -97650x^8 + 2.01e+06x^7 + -2.70291e+07x^6 + 2.47986e+08x^5 + -1.57123e+09x^4 + 6.78506e+09x^3 + -1.91028e+10x^2 + 3.16559e+10x + -2.34469e+10)/3.6288e+06
    [8, 9)p(x) = (9x^10 + -800x^9 + 31950x^8 + -754800x^7 + 1.16781e+07x^6 + -1.23603e+08x^5 + 9.06026e+08x^4 + -4.53956e+09x^3 + 1.4871e+10x^2 + -2.8742e+10x + 2.48715e+10)/3.6288e+06
    [9, 10)p(x) = (-1x^10 + 100x^9 + -4500x^8 + 120000x^7 + -2.1e+06x^6 + 2.52e+07x^5 + -2.1e+08x^4 + 1.2e+09x^3 + -4.5e+09x^2 + 1e+10x + -9.99637e+09)/3.6288e+06
    [10, inf)p(x) = (3.6288e+06)/3.6288e+06
    */
        double v=0;
        if(0<=x && x<1){
            double val=+ xp(10) ;
            v+=val;
        }

        if(1<=x && x<2){
            double val=-10+ 100*xp(1) - 450*xp(2) + 1200*xp(3) - 2100*xp(4) + 2520*xp(5) - 2100*xp(6) + 1200*xp(7) - 450*xp(8) + 100*xp(9) - 9*xp(10) ;
            v+=val;
        }

        if(2<=x && x<3){
            double val=46070- 230300*xp(1) + 517950*xp(2) - 690000*xp(3) + 602700*xp(4) - 360360*xp(5) + 149100*xp(6) - 42000*xp(7) + 7650*xp(8) - 800*xp(9) + 36*xp(10) ;
            v+=val;
        }

        if(3<=x && x<4){
            double val=-7039810+ 23389300*xp(1) - 34911450*xp(2) + 30802800*xp(3) - 17768100*xp(4) + 6987960*xp(5) - 1892100*xp(6) + 346800*xp(7) - 40950*xp(8) + 2800*xp(9) - 84*xp(10) ;
            v+=val;
        }

        if(4<=x && x<5){
            double val=213161150- 527113100*xp(1) + 584403750*xp(2) - 382074000*xp(3) + 162865500*xp(4) - 47202120*xp(5) + 9397500*xp(6) - 1266000*xp(7) + 110250*xp(8) - 5600*xp(9) + 126*xp(10) ;
            v+=val;
        }

        if(5<=x && x<6){
            double val=-2247776350+ 4394761900*xp(1) - 3845283750*xp(2) + 1980426000*xp(3) - 664009500*xp(4) + 151247880*xp(5) - 23677500*xp(6) + 2514000*xp(7) - 173250*xp(8) + 7000*xp(9) - 126*xp(10) ;
            v+=val;
        }

        if(6<=x && x<7){
            double val=10450120610- 16768399700*xp(1) + 12027087450*xp(2) - 5073961200*xp(3) + 1393520100*xp(4) - 260258040*xp(5) + 33476100*xp(6) - 2929200*xp(7) + 166950*xp(8) - 5600*xp(9) + 84*xp(10) ;
            v+=val;
        }

        if(7<=x && x<8){
            double val=-23446909270+ 31655928700*xp(1) - 19102837950*xp(2) + 6785058000*xp(3) - 1571234700*xp(4) + 247985640*xp(5) - 27029100*xp(6) + 2010000*xp(7) - 97650*xp(8) + 2800*xp(9) - 36*xp(10) ;
            v+=val;
        }

        if(8<=x && x<9){
            double val=24871472810- 28742048900*xp(1) + 14871024450*xp(2) - 4539562800*xp(3) + 906026100*xp(4) - 123603480*xp(5) + 11678100*xp(6) - 754800*xp(7) + 31950*xp(8) - 800*xp(9) + 9*xp(10) ;
            v+=val;
        }

        if(9<=x && x<10){
            double val=-9996371200.0000001+ 10000000000*xp(1) - 4500000000*xp(2) + 1200000000*xp(3) - 210000000*xp(4) + 25200000*xp(5) - 2100000*xp(6) + 120000*xp(7) - 4500.0000000000001*xp(8) + 100*xp(9) - 1*xp(10) ;
            v+=val;
        }

        if(10<=x){
            double val=3628800;
            v+=val;
        }

        return v/double(3628800);
    }
    return 0;
}


namespace  {
template<int Degree>
struct fcbs{
    CompoundBoundedPolynomial<Degree> fcbs0=forward_cumulative_extrapolation_basis<Degree>();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-1)> fcbs1=fcbs0.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-2)> fcbs2=fcbs1.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-3)> fcbs3=fcbs2.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-4)> fcbs4=fcbs3.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-5)> fcbs5=fcbs4.derivative();
    double operator()(double time, int derivative) const{
        switch(derivative){
        case 0: return fcbs0(time);
        case 1: return fcbs1(time);
        case 2: return fcbs2(time);
        case 3: return fcbs3(time);
        case 4: return fcbs4(time);
        case 5: return fcbs5(time);
        default:
            mlog()<<"missing derivative: "<<derivative<< " for degree"<<Degree<<"\n";
            return 0;
        }
    }
};



fcbs<0> fcbs0;
fcbs<1> fcbs1;
fcbs<2> fcbs2;
fcbs<3> fcbs3;
fcbs<4> fcbs4;
fcbs<5> fcbs5;
}



double forward_extrapolation_cumulative_cardinal_basis(double time, int degree, int derivative){
    switch(degree)
    {
    case 0: return fcbs0(time,derivative);
    case 1: return fcbs1(time,derivative);
    case 2: return fcbs2(time,derivative);
    case 3: return fcbs3(time,derivative);
    case 4: return fcbs4(time,derivative);
    case 5: return fcbs5(time,derivative);
    default:
        mlog()<<"missing degree: "<<degree<<"\n";
        return 0;
    }
}
namespace  {
template<int Degree>
struct bcbs{
    CompoundBoundedPolynomial<Degree> bcbs0=backwards_cumulative_extrapolation_basis<Degree>();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-1)> bcbs1=bcbs0.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-2)> bcbs2=bcbs1.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-3)> bcbs3=bcbs2.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-4)> bcbs4=bcbs3.derivative();
    CompoundBoundedPolynomial<std::max(0,int(Degree)-5)> bcbs5=bcbs4.derivative();
    double operator()(double time, int derivative) const{
        //static_assert (Degree<6," assumed here, easy fix, but not fast" );
        switch(derivative){
        case 0: return bcbs0(time);
        case 1: return bcbs1(time);
        case 2: return bcbs2(time);
        case 3: return bcbs3(time);
        case 4: return bcbs4(time);
        case 5: return bcbs5(time);
        default: return 0;
        }
    }
};



bcbs<0> bcbs0;
bcbs<1> bcbs1;
bcbs<2> bcbs2;
bcbs<3> bcbs3;
bcbs<4> bcbs4;
bcbs<5> bcbs5;
}
double backward_extrapolation_cumulative_cardinal_basis(double time, int degree, int derivative){
    switch(degree)
    {
    case 0: return bcbs0(time,derivative);
    case 1: return bcbs1(time,derivative);
    case 2: return bcbs2(time,derivative);
    case 3: return bcbs3(time,derivative);
    case 4: return bcbs4(time,derivative);
    case 5: return bcbs5(time,derivative);
    default:
        mlog()<<"missing degree: "<<degree<<"\n";
        return 0;
    }
}

double get_spline_basis_polys_integer_factor(int degree)
{
    if(degree<2) return 1;
    if(degree==2) return 2;
    return degree*get_spline_basis_polys_integer_factor(degree-1);

    //std::array<int,6> fs{1, 1, 2, 6, 24, 120};
}

int SplineBasis::get_first(double time) const {
    return get_last(time)-(degree);
}
int SplineBasis::get_last(double time) const {
    int last=int(std::floor(time / delta_time));
    if(last>last_control_point)
        last=last_control_point;
    if(last<int(first_control_point+degree))
        last=int(first_control_point+degree);
    return last;
}
double SplineBasis::operator()(double time, int index, int derivative) const
{

    double cardinal_time=time/delta_time - index;
    double koeff=cumulative_cardinal_basis(cardinal_time, degree, derivative);

    if(derivative!=0)
        koeff*=std::pow(1.0/delta_time,derivative);
    return koeff;


#if 0
    BasisCoefficients control_point_cumulative_coeffs(double time, int derivative) const {
            BasisCoefficients arr=BasisCoefficients::Zero();
            {
                int j=0;

                for(int i=get_first(time); i <= get_last(time); ++i){
                    double cardinal_time=time/delta_time -i;
                    double koeff=cumulative_cardinal_basis(cardinal_time, degree(), derivative);

                    arr[j++] = koeff*std::pow(1.0/delta_time,derivative);
                }
            }

            // if the last is near the last control point,
            // then we use the forward extrapolation instead

            if(get_last(time)==last_control_point)
            {

                double cardinal_time= time/delta_time- double(last_control_point );
                // note this function is very expensive to compute, and can easily be speed up
                double koeff=forward_extrapolation_cumulative_cardinal_basis(cardinal_time,degree(),derivative);
                koeff*=std::pow(1.0/delta_time,derivative);
                arr[arr.size()-1] =koeff;
                //  cout<<"extrapolate forward "<<arr<<" "<<cardinal_time<<" "<<derivative<<" "<<time<<endl;
            }


            if(get_first(time)==first_control_point ){

                // this is untested, there may be an offset!
                double cardinal_time= time/delta_time - double(first_control_point);
                double koeff=backward_extrapolation_cumulative_cardinal_basis(cardinal_time,degree(),derivative);
                arr[0] = 1;
                if(derivative>0)
                    arr[0]=0;
                arr[1]=  koeff*std::pow(1.0/delta_time,derivative);
                //cout<<"extrapolate backwards "<<arr<<" "<<derivative<<" "<<time<<endl;
            }

            //cout<<"extrapolate backwards: "<<arr<<" "<<arr.norm()<<" "<<get_first(time)<<"\n "<<endl;

            return arr;
        }

#endif





    return 0;}

}
