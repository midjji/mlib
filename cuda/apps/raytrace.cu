#include <opencv2/highgui.hpp>



#include <mlib/cuda/klt/internal/texture.h>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/matrix_adapter.h>
#include <mlib/apps/raytrace/sphere.h>
#include <mlib/apps/raytrace/camera.h>

#include <curand_kernel.h>
#include <mlib/opencv_util/imshow.h>
using std::cout; using std::endl;

int32_t RNG()
{   // mini rng, ?
    unsigned int m_w = 150;
    unsigned int m_z = 40;

    for(int i=0; i < 100; i++)
    {
        m_z = 36969 * (m_z & 65535) + (m_z >> 16);
        m_w = 18000 * (m_w & 65535) + (m_w >> 16);

        return (m_z << 16)+ m_w;  /* 32-bit result */
    }
}









using namespace cvl;
using namespace mlib;
template<class T>
struct World
{
    Tex2<Sphere<T>> spheres;
    __host__ __device__
    bool hit(const Ray<T>& r, float t_min, float t_max, hit_record<T>& rec) const{
        hit_record<T> temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;
        for (int i=0;i<spheres.rows;++i)
        {
            if (spheres(i,0).hit(r, t_min, closest_so_far, temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

};

template<class T, class Random>
__host__ __device__
Vector3<T> cast_ray(Ray<T> r,
                    const World<T>& world,
                    int depth, Random& random)
{
#if 1
    hit_record<T> rec;
    Vector3<T> color, total_attn(1,1,1);
    for(int d=0;d<100;++d)
    {

        if(total_attn.squaredNorm()<0.01) break;
        if (!world.hit(r, 0.001, 1e7, rec))
        {
            Vector3<T> unit_direction = r.direction;
            T t = 0.5*(unit_direction[1] + 1.0);
            return point_multiply(total_attn, (T(1.0f)-t)*Vector3<T>(1.0, 1.0, 1.0) + t*Vector3<T>(0.5, 0.7, 1.0));
        }



        Ray<T> scattered;
        Vector3<T> attenuation;
        if(!rec.mat.scatter(r, rec, attenuation, scattered, random))
        {
            return Vector3<T>(0,0,0);
        }
        total_attn=point_multiply(total_attn, attenuation);
        r=scattered;

    }

    return Vector3<T>(0,0,0);
#else

    hit_record<T> rec;
    if (world.hit(r, 0.001, 1e7, rec))
    {
        Ray<T> scattered;
        Vector3<T> attenuation;
        // 50 for the test
        if (depth < 5 && rec.mat.scatter(r, rec, attenuation, scattered, random))
        {
            return point_multiply(attenuation, cast_ray<T>(scattered, world, depth+1, random));
        }
        return Vector3<T>(0,0,0);
    }

    Vector3<T> unit_direction = r.direction;
    T t = 0.5*(unit_direction[1] + 1.0);
    return (T(1.0f)-t)*Vector3<T>(1.0, 1.0, 1.0) + t*Vector3<T>(0.5, 0.7, 1.0);
#endif
}



namespace kernel
{
__device__
int gidx()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

struct Random
{
    curandState state;
    __device__
    inline float u()
    {
        return curand_uniform(&state);
    }
    __device__
    inline float u(float low, float high){
        return (high - low)*u() + low;
    }
};

template <class T>
__global__ void
raytrace( const World<T> world,
          Tex2<Vector4<T>> image,
          Camera<T> cam
          )
{

    int col = blockIdx.x*blockDim.x  + threadIdx.x;
    int row = blockIdx.y*blockDim.y  + threadIdx.y;
    if(col>=image.cols) return;
    if(row>=image.rows) return;


    Vector4<T> pixel=image(image.rows - row -1,col);
    int cycles=pixel[3];
    Random random;
    curand_init(0, gidx(), cycles, &random.state);





    Vector3<T> color(0,0,0);
    int N=100;
    for(int i=0;i<N;++i)
    {
        // random Ray<T> within the pixel
        T u = T(col + random.u(-0.1,1.1)) / T(image.cols);
        T v = T(row + random.u(-0.1,1.1)) / T(image.rows);
        Ray<T> r = cam.get_ray(u, v);
        color+=cast_ray(r, world,0, random);
    }


    //for(int pi=0;pi<3;++pi)                    pixel[pi]=color[pi]/T(N);




    for(int pi=0;pi<3;++pi)                    pixel[pi]=(pixel[pi]*cycles + color[pi])/T(cycles +N);
    pixel[3]=cycles+N;
    image(image.rows-row-1,col) = pixel;

}

}


inline int divUp( int a, int b )
{
    return (a+b-1)/b;
}

inline float post_process(float x)
{
    x=256.0f*std::sqrt(x);
    if(x<0) x=0;
    if(x>255)x=255;
    return x;
}

void show2(const  Texture<Vector4f, false>& image, std::string name)
{
    cout<<"image: "<<name<< " "<<image.rows()<<" "<<image.cols()<<endl;
    cv::Mat3b rgb(image.rows(), image.cols());
    for(int row=0;row<image.rows();++row)
        for(int col=0;col<image.cols();++col)
        {

            auto x=image(row,col);
            float r=post_process(x[0]);
            float g=post_process(x[1]);
            float b=post_process(x[2]);
            rgb(row,col)=cv::Vec3b(b,g,r);
        }
    imshow(rgb,name);
    cvl::waitKey(30);

}

void raytrace()
{
    using T=float;

    int rows=800;
    int cols=1200;
    Texture<Vector4f, false> himage;
    himage.resize_rc(rows,cols);
    cout<<himage.str()<<endl;

    for(int row=0;row<rows;++row)
        for(int col=0;col<cols;++col)
            himage(row,col)=Vector4f(0.1,0.2,0.3,0);

    Texture<Vector4f, true> dimage;
    dimage=himage;

    mhere()

            std::vector<Sphere<T>> sp=test_scene<T>();
    sp=random_scene<T>(500);
    Texture<Sphere<T>, false> hspheres;
    hspheres.set_to_vec(sp);
    Texture<Sphere<T>, true> dspheres;
    dspheres=hspheres;
    World<T> world{dspheres.tex2()};
    mhere()

            Vector3<T> lookfrom(13,2,3);
    Vector3<T> lookat(0,0,0);
    T dist_to_focus = 10.0;
    T aperture = 0.1;
    Camera<T> cam(lookfrom, lookat, Vector3<T>(0,1,0), 20, T(himage.cols())/T(himage.rows()), aperture, dist_to_focus);

    dim3 dimBlock( 32, 1, 1 ); //
    dim3 dimGrid(  divUp( himage.cols(),  dimBlock.x ),
                   divUp( himage.rows(), dimBlock.y ),
                   1 );

    mhere()
    while(true)
    {

                kernel::raytrace<T><<< dimGrid, dimBlock, 0 >>>(world, dimage.tex2(),cam);
        mhere()
                himage=dimage;
        mhere()
                show2(himage, "post");

    }



}



int main()
{
    raytrace();
    mlib::sleep(1000);
    return 0;

}
