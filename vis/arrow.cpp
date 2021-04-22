#include <mlib/utils/cvl/pose.h>
#include <mlib/vis/arrow.h>
#include <mlib/utils/cvl/lookat.h>

#include <osg/Geometry>


using namespace cvl;
namespace vis {



osg::Node* create_arrow(Flow flow){
    return create_arrow(flow.origin, flow.velocity + flow.origin, flow.color);
}
osg::Node* create_arrow(Vector3d from, Vector3d to, Vector3d color)
{

    osg::ref_ptr<osg::Vec4Array> shared_colors = new osg::Vec4Array;
    shared_colors->push_back(osg::Vec4(color[0],color[1],color[2],0.1f));
    // create Geometry object to store all the vertices and lines primitive.
    osg::Geometry* polyGeom = new osg::Geometry();

    // note, first coord at top, second at bottom, reverse to that buggy OpenGL image..
    std::vector<Vector3f> xs{
        {0,0,0},
        {1,0,0},
        {0,1,0},
        {1,1,0},
        {0,0,1},
        {1,0,1},
        {0,1,1},
        {1,1,1},
        {1,-0.5f,-0.5f},
        {1, 1.5f,-0.5f},
        {1,-0.5f, 1.5f},
        {1, 1.5f, 1.5f},
        {2, 0.5f, 0.5f},
    };

    // origin point!
    for(auto& x:xs)
        x-=Vector3d(0,0.5,0.5);
    // rescale the damned thing, prettier
    //1,3,5,7+
    double len=4;
    for(auto i:std::vector<int>({1,3,5,7,8,9,10,11,12}))
        xs[i]+=Vector3d(len,0,0);

    // rescale the damned again, according to length of the arrow
    // the arrowhead is normally 1/5 of total... so scale the line ? no
    // scale all of it in length direction! i.e. x!

    // general rescaling
    double scale=(from - to).norm()/(xs[0] - xs[12]).norm();
    for(auto& x : xs)
        x*=scale;
    // check the length!
    //cout<<"arrow: "<<(xs[0] - xs[12]).norm()<<" "<<(from - to).norm()<<endl;

    // rotate so x and z swap...
    //double pi=3.14159265359;
    PoseD Pzx(Matrix3d(0,0,-1,0,1,0,1,0,0));//==cvl::getRotationMatrixY(-pi/2.0);



    cvl::PoseD P=cvl::lookAt(to, from, Vector3d(0,1,0)).inverse();
    cvl::PoseD POsg(cvl::Matrix3d(1,0,0,0,-1,0,0,0,-1));
    P=POsg*P;




    for(auto& x:xs)            x=(P*Pzx*x);

    osg::Vec3 myCoords[] =
    {
        // TRIANGLES 6 vertices, v0..v5
        // note in anticlockwise order.
        osg::Vec3(xs[0][0], xs[0][1],xs[0][2]),
        osg::Vec3(xs[1][0], xs[1][1],xs[1][2]),
        osg::Vec3(xs[2][0], xs[2][1],xs[2][2]),
        osg::Vec3(xs[3][0], xs[3][1],xs[3][2]),
        osg::Vec3(xs[4][0], xs[4][1],xs[4][2]),
        osg::Vec3(xs[5][0], xs[5][1],xs[5][2]),
        osg::Vec3(xs[6][0], xs[6][1],xs[6][2]),
        osg::Vec3(xs[7][0], xs[7][1],xs[7][2]), // 7


        // TRIANGLE STRIP 6 vertices, v6..v11
        // note defined top point first,
        // then anticlockwise for the next two points,
        // then alternating to bottom there after.
        osg::Vec3(xs[8][0], xs[8][1],xs[8][2]),
        osg::Vec3(xs[9][0], xs[9][1],xs[9][2]),
        osg::Vec3(xs[10][0], xs[10][1],xs[10][2]),
        osg::Vec3(xs[11][0], xs[11][1],xs[11][2]),
        osg::Vec3(xs[12][0], xs[12][1],xs[12][2]),


        // TRIANGLE FAN 5 vertices, v12..v16
        // note defined in anticlockwise order.


    };

    int numCoords = sizeof(myCoords)/sizeof(osg::Vec3);

    osg::Vec3Array* vertices = new osg::Vec3Array(numCoords,myCoords);

    // pass the created vertex array to the points geometry object.
    polyGeom->setVertexArray(vertices);

    // use the shared color array.
    polyGeom->setColorArray(shared_colors.get(), osg::Array::BIND_OVERALL);


    // use the shared normal array.
    //polyGeom->setNormalArray(shared_normals.get(), osg::Array::BIND_OVERALL);
    {
        unsigned short myIndices[] =            {                  0,1,2,3,6,7,4,5,0,1            };
        int numIndices = sizeof(myIndices)/sizeof(unsigned short);

        // There are three variants of the DrawElements osg::Primitive, UByteDrawElements which
        // contains unsigned char indices, UShortDrawElements which contains unsigned short indices,
        // and UIntDrawElements which contains ... unsigned int indices.
        // The first parameter to DrawElements is
        osg::ref_ptr<osg::DrawElementsUShort> p0=new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLE_STRIP,numIndices,myIndices);
        polyGeom->addPrimitiveSet(p0);
    }
    {
        unsigned short myIndices[] =            {                  8,9,10,11            };
        int numIndices = sizeof(myIndices)/sizeof(unsigned short);
        osg::ref_ptr<osg::DrawElementsUShort> p1=new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLE_STRIP,numIndices,myIndices);
        polyGeom->addPrimitiveSet(p1);
    }
    {
        unsigned short myIndices[] =            {                  12,8,9,10,11            };
        int numIndices = sizeof(myIndices)/sizeof(unsigned short);
        osg::ref_ptr<osg::DrawElementsUShort> p2=new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLE_FAN,numIndices,myIndices);
        polyGeom->addPrimitiveSet(p2);
    }

    return polyGeom;

}
}
