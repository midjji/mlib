#include <iostream>
#include <limits>
#include "mlib/sfm/p3p/p4p.h"
#include <mlib/sfm/p3p/lambdatwist/lambdatwist.p3p.h>


using std::cout;using std::endl;
namespace cvl{














PoseD p4p(const std::vector<cvl::Vector3d>& xs,
          const std::vector<cvl::Vector2d>& yns,
          Vector4<uint> indexes, double max_angle, PoseD reference)
{
    assert(xs.size()==yns.size());
    assert(indexes.size()==4);
    assert(([&]() -> bool{for(uint i:indexes) if (i>=xs.size()) return false; return true;})());




    Vector<cvl::Matrix<double,3,3>,4> Rs;
    Vector<Vector3<double>,4> Ts;

    int valid = p3p_lambdatwist<double,5>(
                yns[indexes[0]].homogeneous(),yns[indexes[1]].homogeneous(),yns[indexes[2]].homogeneous(),
            xs[ indexes[0]],xs[ indexes[1]],xs[ indexes[2]],
            Rs,Ts);



    // check that all of them look ok...
    /*
    for(int v=0;v<valid;++v)
    {
        PoseD P(Rs[v], Ts[v]);
        for(int i=0;i<4;++i){
            double err=((P*xs[indexes[i]]).dehom() - yns[indexes[i]]).squaredNorm();
            std::cout<<"v: "<<v<<" i: "<<i<<" err: "<<err<<std::endl;
        }
    }
    */


    // pick the minimum, whatever it is
    Vector2d y=yns[indexes[3]];
    Vector3d x=xs[indexes[3]];
    PoseD P=PoseD(); // identity
    double e0=std::numeric_limits<double>::max();

    Vector4<bool> actually_valid(false,false,false,false);
    for(int v=0;v<valid;++v)
    {
        // the lambdatwist rotations have a problem with not quite beeing rotations... ???
        // this is extremely uncommon, except when you have very particular kinds of noise,
        // this gets caught by the benchmark somehow?

        Vector4d q=getRotationQuaternion(Rs[v]);
    /*
        p4ptots++;
        if(std::abs(q.length()-1)>1e-3){
            p4pqfails++;
            if(p4pqfails<10 || ((int)p4pqfails) % 1000 == 0)                cout<<"p4p: "<<p4pqfails/p4ptots<<endl;
        }
*/

        q.normalize();
        PoseD tmp(q,Ts[v]);

        if(max_angle<2*3.1415){

            if((tmp*reference.inverse()).angle()>=max_angle) continue;
        }

        Vector3d xr=tmp*x;
        if(xr[2]<=0) continue;
        double e=((Rs[v]*x + Ts[v]).dehom() - y).absSum();
        if (std::isnan(e)) continue;
        if (e<e0 ){
            P=tmp;
            e0=e;
        }
        if(P.is_normal() && e<0.1)
            actually_valid[v]=true;
    }


    return P;
}








}// end namespace cvl
