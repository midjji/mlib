
#include <vector>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/spline/r3.h>
#include <ceres/ceres.h>
#include <mlib/vis/mlib_simple_point_cloud_viewer.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/lookat.h>
#include <mlib/opencv_util/imshow.h>
using namespace cvl;
using std::cout;
using std::endl;


struct GS{
    Vector3f center;
    double sigma;
    double value;
    double factor=value/(sigma*sqrt(2*3.1416));
    inline double operator()(Vector3f x) const
    {
        Vector3f y=x-center;
        y/=sigma;

        return factor*std::exp(-0.5*y.squaredNorm());
    }
    inline Vector3d normal(Vector3f x) const
    {
        const auto& a=*this;
        double k=-a(x)/(sigma*sigma);

        return float(k)*x;
    }

};


struct Map{
    std::vector<GS> gs;
    Map()
    {
        gs.reserve(200);
        gs.push_back(GS{Vector3f(0,0,0), 0.5,1});

        // add a floor
        //for(int a=0;a<10;++a)
        //  for(int b=0;b<10;++b)
        //    gs.push_back(GS{Vector3f(a,b,0),1,5});

        //gs.push_back(GS{Vector3f(0,0,1),4,2});
        //gs.push_back(GS{Vector3f(0,0,2),4,2});
        //gs.push_back(GS{Vector3f(10,10,1),4,3});

    }
    inline double density(Vector3f x) const
    {
        double v=0;
        for(const GS& g:gs)
        {
            v+=g(x);
        }
        v/=double(gs.size());
        return v;
    }
    inline Vector3d normal(Vector3d x) const
    {
        Vector3d n(0,0,0);
        for(const GS& g:gs)
        {
            n+=g.normal(x);
        }
        return n;
    }
};



class DensityCost
{
public:
    Vector3d x; // cardinal cordinates
    double density;
    int degree;

    // if degree is variable it must be a ceres dynamic cost, but that is fine...
    DensityCost(Vector3d x, double density, int degree):x(x), density(density), degree(degree){}
    template <typename T>
    bool operator()(const T* const spline_local_window,
                    T* residuals) const
    {
        residuals[0]=T(density)  - spline3d(spline_local_window, x, degree);
        return true;
    }
};
/*
ceres::CostFunction* density_cost(Vector3d x, double density, Spline s)
{

    ceres::DynamicAutoDiffCostFunction<DensityCost, 4>* cost_function =
            new ceres::DynamicAutoDiffCostFunction<DensityCost, 4>(
                new DensityCost(x, density, s.degree()));

    cost_function->AddParameterBlock(s.local_window_size());
    //cost_function->AddParameterBlock(10);
    cost_function->SetNumResiduals(1);


    //return (new ceres::AutoDiffCostFunction<DensityCost, 1, s.local_window_size()>(new DensityCost(x, density, degree)));
}
*/

struct Ray{
    Vector3d origin;
    Vector3d direction;
};
Ray operator*(PoseD Pwc,  Ray ray){
    Ray out;
    out.origin = Pwc*ray.origin;
    out.direction = Pwc.rotate(ray.direction);
    return out;
}

struct Camera
{
    int rows=100;
    int cols=100;
    double fx=50;
    double fy=50;
    double px=50;
    double py=50;
    Vector2d project(Vector3d x)
    {
        return Vector2d(x[1]*fy/x[2] +py,x[0]*fx/x[2] +px);
    }
    Vector3d unproject(double row, double col)
    {
        return Vector3d((col -px)/fx, (row-py)/fy, 1);
    }
    Ray ray(PoseD Pwc, double row, double col){
        Ray ray{Vector3d(0,0,0), unproject(row, col)};
        return Pwc*ray;
    }
};


bool hit(const Map& map, const Ray& ray_w, Vector3d& x, double min_density)
{

    for(double t=1;t<20;t+=0.01)
    {
        x=ray_w.origin + t*ray_w.direction;
        double d=map.density(x);
        if(d>min_density)
            return true;
    }
    return false;
}

Vector3d cast_ray(const Ray& r,
                  const Map& map, double& distance, const std::vector<Vector3d>& lights)
{




    double diffuse_k   = 0.7;

    Vector3d intensity_m(1,1,1);
    Vector3d intensity_m_s(1,1,1);

    Vector3d X;distance=10;

    if (!hit(map, r, X, 0.01))
    {
        return Vector3d(0,0,0);
    }
    distance=X.norm();

    Vector3f ambient = Vector3d(0,0,0); // ki*ai
    Vector3f color= ambient;
    for(auto light:lights)
    {
        Vector3d L=(light - X).normalized();
        Vector3d N=map.normal(X).normalized();

        double dotproduct=(L.dot(N));
        if(dotproduct>0)
        {
            Vector3d diffuse=diffuse_k*dotproduct*intensity_m;
           //color+=diffuse;
        }


        double specular_k      = 0.3;
        double specular_alpha  = 20;
        Vector3d R=(2.0*(L.dot(N))*N - L).normalized();
        Vector3d V= X.normalized();
        double specular=R.dot(V);
        if(specular>0)
        {

            color+=std::pow(specular,specular_alpha)*specular_k*intensity_m_s;
        }


    }
    return color;
}


void show_map2(Map map, cv::Mat3f image,  Camera camera, PoseD Pwc, cv::Mat1f dist_image)
{


    std::vector<Vector3d> lights;
    lights.push_back(Vector3d(-100,0,-75));
    //lights.push_back(Vector3d(-10,0,-3));

    mlib::Timer timer;
    for (int row = 0; row < image.rows; ++row)
    {
        for (int col = 0; col < image.cols; ++col)
        {

            timer.tic();
            Ray ray=camera.ray(Pwc, row, col);
            double distance;
            Vector3d color=cast_ray(ray, map, distance, lights);
            cv::Vec3f& pixel=image(row,col);
            for(int pi=0;pi<3;++pi)                    pixel[pi]=color[pi];
            timer.toc();

            dist_image(row,col)=distance;
        }
        cout<<timer<<endl;
    }






}



void show_map(Map map)
{

    double w=10;
    double delta=0.2;
    int N=(w*2+1)/delta;
    N=N*N*N;
    std::vector<Vector3d> xs;xs.reserve(N);
    std::vector<mlib::Color> cs;cs.reserve(N);
    for(double a=-w;a<w;a+=0.1)
        for(double b=-w;b<w;b+=0.1)
            for(double c=-w;c<w;c+=0.1)
            {
                xs.push_back(Vector3d(a,b,c));
                cs.push_back(mlib::jet(map.density(Vector3d(a,b,c)),0,0.4));
            }
    mlib::pc_viewer("map")->setPointCloud(xs,cs);

}


template<class T>
inline uchar cap(T v)
{
    if(v<0.0f) return 0;
    if(v<255.0f) return v;
    return 255;
}
cv::Mat3b parse(cv::Mat3f& image)
{
    cv::Mat3b rgb(image.rows, image.cols);
    for(int r=0;r<image.rows;++r)
        for(int c=0;c<image.cols;++c)
        {
            auto& opx=rgb(r,c);
            auto& ipx=image(r,c);



            opx[0] = cap(256.0f*std::sqrt(ipx[2]));
            opx[1] = cap(256.0f*std::sqrt(ipx[1]));
            opx[2] = cap(256.0f*std::sqrt(ipx[0]));

        }
    return rgb;
}

cv::Mat3b distance2rgb(cv::Mat1f& image)
{
    cv::Mat3b rgb(image.rows, image.cols);
    for(int r=0;r<image.rows;++r)
        for(int c=0;c<image.cols;++c)
            rgb(r,c)=mlib::jet(image(r,c),0,5).fliprb().toV3<cv::Vec3b>();
    return rgb;

}

cv::Mat1b distance2grey(cv::Mat1f& image)
{
    cv::Mat1b rgb(image.rows, image.cols);
    for(int r=0;r<image.rows;++r)
        for(int c=0;c<image.cols;++c)
            rgb(r,c)=cap(255.0*((image(r,c)-0.9)/1.5));
    return rgb;
}

int main()
{
    // old vec class, 42, now 72
    // pre matrix refactor 72, now 56,
    // pre material pointer refactor 56, now 45

    Camera cam;

    cv::Mat3f image(cam.rows,cam.cols);
    cv::Mat1f dist_image(cam.rows, cam.cols);
    Map map;
    // look from x=0, y-3, z=-5;M
    PoseD Pwc=lookAt_sane(Vector3d(0,0,0), Vector3d(0,0,-3), Vector3d(0,-1,0)).inverse();




    std::thread thr([&](){
        show_map(map);
        show_map2(map, image,cam, Pwc, dist_image);
    });

    cv::Mat3b rgb(image.rows, image.cols);
    while(true)
    {
        cvl::imshow(parse(image),"shaded density");
        cvl::imshow(distance2rgb(dist_image),"distance image (jet)");
        cvl::imshow(distance2grey(dist_image),"distance image");
        cvl::imshow(10*dist_image,"distance image 2");
        waitKey(0);
    }
    thr.join();
}
