#include <kitti/mots/calibration.h>
#include <filesystem>



namespace cvl {

KittiMotsStereoCalibration KittiMotsCalibration::get_color_stereo(std::string basepath,
                                                   bool training,
                                                   int sequence){
    KittiMotsCalibration kmc;
    kmc.read_file(mots_calib_path(basepath, training, sequence));
    return KittiMotsStereoCalibration{kmc.fx(), kmc.fy(),kmc.px(),kmc.py(),kmc.color_baseline()};
}



    std::string KittiMotsCalibration::read_chars(std::ifstream& ifs, int num){
        std::string ret;ret.reserve(num);
        for(int i=0;i<num;++i){
            char key;
            ifs>>key;
            ret.push_back(key);
        }

        return ret;
    }
    void KittiMotsCalibration::read_check_test(std::ifstream& ifs, std::string expected, std::string path){
        auto s=read_chars(ifs,expected.size());
        if(s!=expected)
            std::cout<<"calibration failed! expected \""<<expected<<"\" got "      <<s<<" from: "<<path<<std::endl;
        if(!ifs)
            std::cout<<"calibration failed! expected \""<<expected<<"\" got !ifs " <<path<<std::endl;
    }
    std::vector<double> KittiMotsCalibration::read_doubles(std::ifstream& ifs, int num){
        std::vector<double> ds;ds.reserve(num);
        for(int i=0;i<num;++i){
            double d;
            ifs>>d;
            ds.push_back(d);
        }
        if(!ifs)
            std::cout<<"calibration failed! when reading doubles, got !ifs"<<std::endl;

        return ds;
    }

    bool KittiMotsCalibration::test_sanity(std::vector<double> ds){
        if(ds.size()!=12) std::cout<<"wtf?"<<ds.size()<<std::endl;
        if(ds[1]!=0) return false;
        //if(ds[3]!=0) return false;
        if(ds[4]!=0) return false;
        //if(ds[7]!=0) return false;
        if(ds[8]!=0) return false;
        if(ds[9]!=0) return false;
        //if(ds[11]!=0) return false;
        return true;
    }
    bool KittiMotsCalibration::test_common_K(){
        if(p0s[0]!=p1s[0]) return false;
        if(p0s[0]!=p2s[0]) return false;
        if(p0s[0]!=p3s[0]) return false;

        if(p0s[2]!=p1s[2]) return false;
        if(p0s[2]!=p2s[2]) return false;
        if(p0s[2]!=p3s[2]) return false;

        if(p0s[6]!=p1s[6]) return false;
        if(p0s[6]!=p2s[6]) return false;
        if(p0s[6]!=p3s[6]) return false;

        if(p0s[5]!=p1s[5]) return false;
        if(p0s[5]!=p2s[5]) return false;
        if(p0s[5]!=p3s[5]) return false;


        return true;
    }
    bool KittiMotsCalibration::shared_k(KittiMotsCalibration km2){
            for(int i=0;i<12;++i){
         if(p0s[i]!=km2.p0s[i]) return false;
         if(p1s[i]!=km2.p1s[i]) return false;
         if(p2s[i]!=km2.p2s[i]) return false;
         if(p3s[i]!=km2.p3s[i]) return false;
            }
            return true;
    }
    bool KittiMotsCalibration::read_file(const std::string& path){

        /**
P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00
    0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00
    0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02
    0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00
    0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01
    0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01
    0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02
    0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00
    0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
R_rect   9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03
        -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03
         7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
Tr_velo_cam 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_imu_velo 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
          */
        // step 0 read them raw
        // remember that << skips whitespace...
        if(!std::filesystem::exists(std::filesystem::path(path))){
            std::cout<<"calibration not found!"<<path<<std::endl;
            exit(1);
        }
        std::ifstream ifs(path);

        read_check_test(ifs,"P0:", path);
        p0s=read_doubles(ifs,12);


        read_check_test(ifs,"P1:", path);
        p1s=read_doubles(ifs,12);


        read_check_test(ifs,"P2:", path);
        p2s=read_doubles(ifs,12);


        read_check_test(ifs,"P3:", path);
        p3s=read_doubles(ifs,12);


        read_check_test(ifs,"R_rect", path);
        r0s=read_doubles(ifs,9);
        read_check_test(ifs,"Tr_velo_cam", path);
        tvs=read_doubles(ifs,12);
        read_check_test(ifs,"Tr_imu_velo", path);
        trs=read_doubles(ifs,12);


        if(!test_sanity(p0s)) return false;
        if(!test_sanity(p1s)) return false;
        if(!test_sanity(p2s)) return false;
        if(!test_sanity(p3s)) return false;

        if(!test_common_K()) return false;

        return true;
    }

std::string mots_calib_path(std::string basepath, bool training, int sequence){
    std::string path=basepath+"calib/";
    if(training)
        path+="training/";
    else
        path+="testing/";
    path+="calib/"+mlib::toZstring(sequence,4)+".txt";
    return path;
}
bool KittiMotsCalibration::test_common_translations(KittiMotsCalibration km2){

    std::cout<<translation(p3s) - translation(p2s)<<std::endl;
    std::cout<<translation(km2.p3s) - translation(km2.p2s)<<std::endl;
/*
    if((translation(p0s) - translation(km2.p0s)).norm()>0)
        std::cout<<"diff0: "<<translation(p0s) <<" "<< translation(km2.p0s)<<std::endl;

    if((translation(p1s) - translation(km2.p1s)).norm()>0)
        std::cout<<"diff1: "<<translation(p1s) <<" "<< translation(km2.p1s)<<std::endl;

    if((translation(p2s) - translation(km2.p2s)).norm()>0)
        std::cout<<"diff2: "<<translation(p2s) <<" "<< translation(km2.p2s)<<std::endl;

    if((translation(p3s) - translation(km2.p3s)).norm()>0)
        std::cout<<"diff3: "<<translation(p3s) <<" "<< translation(km2.p3s)<<std::endl;
*/
    if(p0s[3]!=km2.p0s[3]) return false;
    if(p0s[7]!=km2.p0s[7]) return false;

    if(p1s[3]!=km2.p1s[3]) return false;
    if(p1s[7]!=km2.p1s[7]) return false;

    if(p2s[3]!=km2.p2s[3]) return false;
    if(p2s[7]!=km2.p2s[7]) return false;
    if(p3s[3]!=km2.p3s[3]) return false;
    if(p3s[7]!=km2.p3s[7]) return false;

    return true;

}
}
