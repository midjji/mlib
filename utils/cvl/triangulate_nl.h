#pragma once
#include <mlib/utils/cvl/pose.h>
#include <vector>
#include <mlib/utils/cvl/triangulate.h>
namespace cvl{

template<class T,int Samples> Vector<T,Samples> get_vector(std::vector<T>& ts){
    assert(ts.size()==Samples);
    Vector<T,Samples> t;
    for(uint i=0;i<Samples;++i)
        t[i]=ts[i];
    return t;
}

template<int Samples>
Matrix<double,2*Samples,3> get_J(Vector<Matrix4d,Samples>& ps,
                                 Vector3d x0){
    Matrix<double,2*Samples,3> J;
    for(uint i=0;i<Samples;++i) {

        // for y0
        // y-((aj + m0jxj)/(aj + m2jxj)) deriv map xj is
        // m0j
        Matrix4d M=ps[i];
        Vector3d xt=M*x0;
        double dxt0dx0 = M(0,0);        double dxt0dx1 = M(0,1);        double dxt0dx2 = M(0,2);
        double dxt1dx0 = M(1,0);        double dxt1dx1 = M(1,1);        double dxt1dx2 = M(1,2);
        double dxt2dx0 = M(2,0);        double dxt2dx1 = M(2,1);        double dxt2dx2 = M(2,2);
        // first for y0,
        // derivative with respect to x_i
        double xt22=(xt[2]*xt[2]); if(xt22<1e-9) xt22=1e-9;
        //double a00=M(0,1)*x0[1] + M(0,2)*x0[2] + M(0,3);
        //double a20=M(2,1)*x0[1] + M(2,2)*x0[2] + M(2,3);
        //double test00=(M(0,0)*a20 - M(2,0)*a00)/(xt22);

        double dr0dx0=  (dxt0dx0*xt[2] - xt[0]*dxt2dx0)/(xt22);
        //double test=(M(0,0)*a20 - M(2,0)*a00)/(xt22);
        //cout<<"test: "<<test<< " "<<dr0dx0<< " "<<test-dr0dx0<<endl;
        double dr0dx1=  (dxt0dx1*xt[2] - xt[0]*dxt2dx1)/(xt22);
        double dr0dx2=  (dxt0dx2*xt[2] - xt[0]*dxt2dx2)/(xt22);
        J(2*i,0)=dr0dx0;        J(2*i,1)=dr0dx1;        J(2*i,2)=dr0dx2;

        // derivative with respect to x_i
        double dr1dx0=  (dxt1dx0*xt[2] - xt[1]*dxt2dx0)/(xt22);
        double dr1dx1=  (dxt1dx1*xt[2] - xt[1]*dxt2dx1)/(xt22);
        double dr1dx2=  (dxt1dx2*xt[2] - xt[1]*dxt2dx2)/(xt22);
        J(2*i+1,0)=dr1dx0;        J(2*i+1,1)=dr1dx1;        J(2*i+1,2)=dr1dx2;
    }
    return J;
}


template<int Samples>
Matrix<double,2*Samples,1>
get_r(Vector<Vector2d,Samples>& ys,
      Vector<Matrix4d,Samples>& ps,
      Vector3d x0){
    Matrix<double,2*Samples,1> r;
    for(uint i=0;i<Samples;++i) {
        Vector2d yr=(ps[i]*x0).dehom();
        r[2*i]=    -yr[0] + ys[i][0];
        r[2*i+1]=  -yr[1] + ys[i][1];
    }
    return r;
}



template<int Samples>
Vector3d gn_minimize(Vector<Vector2d,Samples> ys,
                     Vector<Matrix4d,Samples> ps,
                     Vector3d x0){

    Vector3d x=x0;
    Matrix<double,2*Samples,1> r = get_r<Samples>(ys,ps,x);
    double error=r.squaredNorm();

    for(uint i=0;i<5;++i){
        if(error<1e-10) break;
        Matrix<double,2*Samples,3> J =get_J<Samples>(ps,x);// dr(i)dx
        Vector3d g=J.transpose()*r;
        Matrix<double,3,3> A=J.transpose()*J;
        Vector3d v=A.inverse()*g; // num suboptimal, but not bad...
        if(v.absMax()<1e-12) break;
        Matrix<double,2*Samples,1> nr;
        double nerror;
        Vector3d x_test;
        double f=1;
        while(true){
            x_test=x+f*v;
            nr=get_r<Samples>(ys,ps,x_test);
            nerror=nr.squaredNorm();
            if(nerror<error && x_test.isnormal()){
                r=nr;
                error=nerror;
                x=x_test;
                break;
            }
            else{
                f*=0.5;
                if(f<0.5*0.5*0.5*0.5*0.5*0.5){
                    i+=10;
                    break;
                }
                i+=1;
            }
        }
    }

    {// check we didnt make it worse!
        Matrix<double,2*Samples,1> r0 = get_r<Samples>(ys,ps,x0);
        double error0=r0.squaredNorm();
        Matrix<double,2*Samples,1> r1 = get_r<Samples>(ys,ps,x);
        double error=r1.squaredNorm();
        if(error0<error){
            std::cout<<"made it worse"<<std::endl;
            return x0;
        }
    }
    return x;
}

template<int Samples>
Vector3d gn_minimize(std::vector<Vector2d>& ys,
                     std::vector<PoseD>& ps,
                     Vector3d x0){
    std::vector<Matrix4d> ms;ms.reserve(ps.size());
    for(PoseD& p:ps)
        ms.push_back(p.get4x4());
    return gn_minimize<Samples>(get_vector<Vector2d,Samples>(ys),get_vector<Matrix4d,Samples>(ms),x0);
}


template<class T>
Vector3<T> gn_minimize(std::vector<Vector2<T>>& ys, std::vector<Pose<T>>& ps, Vector3<T> x0){
    assert(ys.size()==ps.size());
    switch (ys.size()) {
    case 2: return gn_minimize<2>(ys,ps,x0);
    case 3: return gn_minimize<3>(ys,ps,x0);
    case 4: return gn_minimize<4>(ys,ps,x0);
    case 5: return gn_minimize<5>(ys,ps,x0);
    case 6: return gn_minimize<6>(ys,ps,x0);
    case 7: return gn_minimize<7>(ys,ps,x0);
    case 8: return gn_minimize<8>(ys,ps,x0);
    case 9: return gn_minimize<9>(ys,ps,x0);
    case 10: return gn_minimize<10>(ys,ps,x0);
    case 11: return gn_minimize<11>(ys,ps,x0);
    case 12: return gn_minimize<12>(ys,ps,x0);
    case 13: return gn_minimize<13>(ys,ps,x0);
    case 14: return gn_minimize<14>(ys,ps,x0);
    case 15: return gn_minimize<15>(ys,ps,x0);
    case 16: return gn_minimize<16>(ys,ps,x0);
    case 17: return gn_minimize<17>(ys,ps,x0);
    case 18: return gn_minimize<18>(ys,ps,x0);
    case 19: return gn_minimize<19>(ys,ps,x0);
    case 20: return gn_minimize<20>(ys,ps,x0);
    case 21: return gn_minimize<21>(ys,ps,x0);
    case 22: return gn_minimize<22>(ys,ps,x0);
    case 23: return gn_minimize<23>(ys,ps,x0);
    case 24: return gn_minimize<24>(ys,ps,x0);
    case 25: return gn_minimize<25>(ys,ps,x0);
    case 26: return gn_minimize<26>(ys,ps,x0);
    case 27: return gn_minimize<27>(ys,ps,x0);
    case 28: return gn_minimize<28>(ys,ps,x0);
    case 29: return gn_minimize<29>(ys,ps,x0);
    case 30: return gn_minimize<30>(ys,ps,x0);
    case 31: return gn_minimize<31>(ys,ps,x0);
    case 32: return gn_minimize<32>(ys,ps,x0);
    case 33: return gn_minimize<33>(ys,ps,x0);
    case 34: return gn_minimize<34>(ys,ps,x0);
    case 35: return gn_minimize<35>(ys,ps,x0);
    case 36: return gn_minimize<36>(ys,ps,x0);
    case 37: return gn_minimize<37>(ys,ps,x0);
    case 38: return gn_minimize<38>(ys,ps,x0);
    case 39: return gn_minimize<39>(ys,ps,x0);
    case 40: return gn_minimize<40>(ys,ps,x0);
    case 41: return gn_minimize<41>(ys,ps,x0);
    case 42: return gn_minimize<42>(ys,ps,x0);
    case 43: return gn_minimize<43>(ys,ps,x0);
    case 44: return gn_minimize<44>(ys,ps,x0);
    case 45: return gn_minimize<45>(ys,ps,x0);
    case 46: return gn_minimize<46>(ys,ps,x0);
    case 47: return gn_minimize<47>(ys,ps,x0);
    case 48: return gn_minimize<48>(ys,ps,x0);
    case 49: return gn_minimize<49>(ys,ps,x0);
    case 50: return gn_minimize<50>(ys,ps,x0);
    default:
        assert(false && "bad size!");
        return gn_minimize<50>(ys,ps,x0);
    }
}
template<class T>
Vector3<T> triangulate_nl(std::vector<Vector2<T>> ys, std::vector<Pose<T>> ps){
    assert(ys.size()==ps.size());
    assert(ys.size()>1);
    Vector3<T> x0=triangulate(PoseD(ps[0]),PoseD(ps[1]),Vector2d(ys[0]),Vector2d(ys[1]));
    return gn_minimize(ys,ps,x0);
}
}
