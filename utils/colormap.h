#pragma once
#include <stdint.h>
#include <vector>
#include <mlib/utils/cvl/matrix.h>

namespace mlib{


/**
 * @brief The Color class
 * cv::scalar as color wrapper...
 * random and ordered distinct colors
 * ints encode colors assumed between 0 and 255, change to byte? perhaps later...
 */
class Color{
public:

    Color();
    // 0xrrggbb00
    Color(uint32_t rgb);

    Color(uint8_t r, uint8_t g, uint8_t b);

    // replace with only good ones
    static Color red();
    static Color green();
    static Color blue();
    static Color cyan();
    static Color white();
    static Color black();
    static Color gray();
    static Color pink();
    static Color random();
    static Color yellow();
    // numbered bright distinct colors: modulo count 7
    static Color nr(int i);
    // numbered bright distinct colors: modulo count (larger 30 ish)
    static Color nrlarge(int i);
    static Color next();
    // colors are in rgb order and assume rgb images, so this tosses it to bgr for opencv
    Color fliprb() const;
    cvl::Vector3d cvl();

    static Color codeDepthRedToDarkGreen(double depth, double mindepth=1, double maxdepth=100);
    int getR() const;
    int getG() const;
    int getB() const;
    // member variable access    
    int& operator[](int i );
    const int& operator[](int i ) const;

    // for opencv scalar without including it as a dependency...
    //cv::Scalar toScalar(){return cv::Scalar(rgb[0],rgb[1],rgb[2],0);}
    template<class T> T toScalar(){return T(rgb[0],rgb[1],rgb[2],0);}
private: 
        int rgb[3];
        static int counter;
};

Color gray2jet(uint8_t graylevel);

/**
 * @brief encodeRed2DarkGreen
 * @param f_value_d
 * @param f_min_d
 * @param f_max_d
 * @param r
 * @param g
 * @param b
 *
 * red to dark green color span
 */
void encodeRed2DarkGreen(  const double  f_value_d,
                           const double  f_min_d,
                           const double  f_max_d,
                           int&          r,
                           int&          g,
                           int&          b);


std::vector<cvl::Vector3d> color2cvl(const std::vector<Color>& color);

}// end namespace mlib
