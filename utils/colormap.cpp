#include "mlib/utils/colormap.h"
#include <mlib/utils/random.h>

#include "colormap_tables/jet_colormap.h"

typedef unsigned char uchar;
namespace mlib {

inline void cap(int& v, int low = 0, int high = 255)
{
    if (v < low) v = low;
    if (v > high) v = high;
}

Color gray2jet(uint8_t graylevel){
   uint8_t r = colormaps::jet[graylevel*3 + 0]; 
   uint8_t g = colormaps::jet[graylevel*3 + 1];
   uint8_t b = colormaps::jet[graylevel*3 + 2];

   Color c = Color(r, g, b);

   return c;
}

void encodeRed2DarkGreen(const double  f_value_d,
                         const double  f_min_d,
                         const double  f_max_d,
                         int&          r,
                         int&          g,
                         int&          b)
{
    double dX = (f_value_d - f_min_d) / (f_max_d - f_min_d);
    r = 0;
    g = 0;
    b = 0; // Blue is always zero

    if (dX > 1.0) {
        g = (int)(382.5 - dX * 127.5);
        cap(g, 63, 255);
        return;
    }
    if (dX < 0.0) {
        r = 255;
        return;
    }
    g = int(510.0 * dX);
    r = int((-510.0 * dX) + 510.0);
    cap(r, 0, 255);
    cap(g, 0, 255);
}


Color::Color()
{
    rgb[0] = rgb[1] = rgb[2] = 0;
}

Color::Color(uint32_t rgb)
{
    // bitshift
    uint8_t r=(uint8_t) (rgb >>24);
    uint8_t g=(uint8_t) (rgb >>16);
    uint8_t b=(uint8_t) (rgb >>8);
    this->rgb[0] = r;
    this->rgb[1] = g;
    this->rgb[2] = b;
   // fliprb();
}
std::vector<cvl::Vector3d> color2cvl(const std::vector<Color>& color){
    std::vector<cvl::Vector3d> rets;
    rets.reserve(color.size());
    for(auto c:color)
        rets.push_back(c.cvl());
}

Color::Color(uint8_t r, uint8_t g, uint8_t b):rgb{r,g,b}
{

}

Color Color::red() { return Color(255, 0, 0); }
Color Color::green() { return Color(0, 255, 0); }
Color Color::blue() { return Color(0, 0, 255); }
Color Color::cyan() { return Color(0, 255, 255); }
Color Color::white() { return Color(255, 255, 255); }
Color Color::black() { return Color(0, 0, 0); }
Color Color::gray() { return Color(128, 128, 128); }
Color Color::pink() { return Color(255, 130, 171); }
Color Color::yellow() { return Color(255, 255, 0); }

Color Color::random()
{
    auto x=random::random_unit_vector<3>();
    // relies on wraparound
    return Color(uint8_t(x[0]*255), uint8_t(x[1]*255), uint8_t(x[2]*255));
}
namespace  {


std::vector<Color> colors = {
    Color::red(),
    Color::green(),
    Color::blue(),
    Color::yellow(),
    Color::cyan(),
    Color::white(),
    //Color::gray(),
    Color::pink(),
    //Color::random(),
};

std::vector<Color> distinctColors={

    //#000000// black is reserved!

    Color(0xFF000000),
    Color(0x00FF0000),
    Color(0x0000FF00),
    Color(0xFFFFFF00),
    Color(0x01FFFE00),
    Color(0xFFA6FE00),
    Color(0xFFDB6600),
    Color(0x00640100),
    Color(0x01006700),
    Color(0x95003A00),
    Color(0x007DB500),
    Color(0xFF00F600),
    Color(0xFFEEE800),
    Color(0x774D0000),
    Color(0x90FB9200),
    Color(0x0076FF00),
    Color(0xD5FF0000),
    Color(0xFF937E00),
    Color(0x6A826C00),
    Color(0xFF029D00),
    Color(0xFE890000),
    Color(0x7A478200),
    Color(0x7E2DD200),
    Color(0x85A90000),
    Color(0xFF005600),
    Color(0xA4240000),
    Color(0x00AE7E00),
    Color(0x683D3B00),
    Color(0xBDC6FF00),
    Color(0x26340000),
    Color(0xBDD39300),
    Color(0x00B91700),
    Color(0x9E008E00),
    Color(0x00154400),
    Color(0xC28C9F00),
    Color(0xFF74A300),
    Color(0x01D0FF00),
    Color(0x00475400),
    Color(0xE56FFE00),
    Color(0x78823100),
    Color(0x0E4CA100),
    Color(0x91D0CB00),
    Color(0xBE997000),
    Color(0x968AE800),
    Color(0xBB880000),
    Color(0x43002C00),
    Color(0xDEFF7400),
    Color(0x00FFC600),
    Color(0xFFE50200),
    //Color(0x620E0000),
    Color(0x008F9C00),
    Color(0x98FF5200),
    Color(0x7544B100),
    Color(0xB500FF00),
    Color(0x00FF7800),
    Color(0xFF6E4100),
    Color(0x005F3900),
    Color(0x6B688200),
    Color(0x5FAD4E00),
    Color(0xA7574000),
    Color(0xA5FFD200),
    Color(0xFFB16700),
    Color(0x009BFF00),
    Color(0xE85EBE00)

};

std::vector<Color> brights={
    Color(255,0,0),
    Color(0,255,0),
    Color(0,0,255),
    ///Color(255,255,255),
    Color(1,255,254),
    Color(255,166,254),
    Color(255,219,102),
    //Color(0,100,1),
    //Color(1,0,103),
    Color(149,0,58),
    //Color(0,125,181),
    Color(255,0,246),
    //Color(255,238,232),
    Color(119,77,0),
    Color(144,251,146),
    Color(0,118,255),
    Color(213,255,0),
    Color(255,147,126),
    //Color(106,130,108),
    Color(255,2,157),
    Color(254,137,0),
    //Color(122,71,130),
    Color(126,0,210),
    Color(133,169,0),
    Color(255,0,86),
    Color(164,36,0),
    Color(0,174,126),
    //Color(104,61,59),
    Color(189,198,255),
    //Color(38,52,0),
    Color(189,211,147),
    Color(0,185,23),
    Color(158,0,142),
    //Color(0,21,68),
    //Color(194,140,159),
    Color(255,116,163),
    Color(1,208,255),
    //Color(0,71,84),
    Color(229,111,254),
    //Color(120,130,0),
    //Color(14,76,161),
    Color(145,208,203),
    Color(190,153,112),
    Color(150,138,232),
    Color(187,136,0),
//    Color(67,0,44),
    Color(222,255,116),
    Color(0,255,198),
    Color(255,229,2),
    Color(0,143,156),
    Color(152,255,82),
    Color(117,68,177),
    Color(181,0,255),
    Color(0,255,120),
    Color(255,110,65),
    //Color(0,95,57),
    //Color(107,104,130),
    Color(95,173,78),
    Color(167,87,64),
    Color(165,255,210),
    Color(255,177,103),
    Color(0,155,255),
    Color(232,94,190),
};
}

Color Color::nr(int i)
{
    return colors.at(i%colors.size());
}
Color Color::nrlarge(int i)
{
    return brights.at(i%brights.size());
}

int Color::counter = 0;

Color Color::next()
{
    return Color::nr(counter++);
}
Color Color::fliprb() const{return Color((std::uint8_t)rgb[2],(std::uint8_t)rgb[1],(std::uint8_t)rgb[0]);}
cvl::Vector3d Color::cvl(){return cvl::Vector3d(rgb[0],rgb[1],rgb[2]);}
Color Color::codeDepthRedToDarkGreen(double depth, double mindepth, double maxdepth)
{
    if(depth<mindepth)
        depth=mindepth;
    if(depth>maxdepth)
        depth=maxdepth;
    int b, g, r;
    encodeRed2DarkGreen(depth, mindepth, maxdepth, r, g, b);
    return Color((uint8_t)r, (uint8_t)g, (uint8_t)b);
}

int Color::getR()  const
{
    return rgb[0];
}

int Color::getG() const
{
    return rgb[1];
}

int Color::getB() const
{
    return rgb[2];
}
// member variable access
int& Color::operator[]( int i )    { return rgb[i]; }
const int& Color::operator[]( int i ) const    {return rgb[i]; }

}// end namespace mlib
