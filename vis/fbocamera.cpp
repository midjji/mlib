#include <mlib/vis/CvGL.h>
#include <Eigen/Eigen>
#include <mlib/vis/fbocamera.h>

FBOCamera::FBOCamera(const int width, const int height, const cvl::Matrix3d& K)
{

    // Create the texture.

    texture = new osg::Texture2D;

    texture->setTextureSize(width, height);
    texture->setInternalFormat(GL_RGBA);
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);

    // Create the framebuffer object.

    fbo = new osg::FrameBufferObject();

    fbo->setAttachment(osg::Camera::COLOR_BUFFER, osg::FrameBufferAttachment(texture));
    fbo->setAttachment(osg::Camera::DEPTH_BUFFER, osg::FrameBufferAttachment(
                           new osg::RenderBuffer(width, height, GL_DEPTH_COMPONENT24)));

    // Create the camera.

    osgCamera = new osg::Camera;
    osgCamera->setClearColor(osg::Vec4(0.1f,0.1f,0.3f,1.0f));
    osgCamera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    osgCamera->setViewport(0, 0, width, height);
    osgCamera->setRenderOrder(osg::Camera::PRE_RENDER);	// render before the main camera

    osg::Matrixd pmat = CvKToGlProjection(K, width, height, 0.1f, 100);
    osgCamera->setProjectionMatrix(pmat);

    osgCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    osgCamera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    osgCamera->attach(osg::Camera::COLOR_BUFFER, texture);
}

/** Create the quad that will display the camera's viewport */
osg::Node *FBOCamera::CreateQuad(const float width, const float height)
{
    osg::StateSet *state;

    osg::Geometry *quad = new osg::Geometry;
    osg::Vec3Array *vx = new osg::Vec3Array;

    const float hw = width / 2;
    const float hh = height / 2;

    vx->push_back(osg::Vec3(-hw, 0, -hh));
    vx->push_back(osg::Vec3( hw, 0, -hh));
    vx->push_back(osg::Vec3( hw, 0,  hh));
    vx->push_back(osg::Vec3(-hw, 0,  hh));
    quad->setVertexArray(vx);

    osg::Vec2Array *tx = new osg::Vec2Array;
    tx->push_back(osg::Vec2(0, 0));
    tx->push_back(osg::Vec2(1, 0));
    tx->push_back(osg::Vec2(1, 1));
    tx->push_back(osg::Vec2(0, 1));
    quad->setTexCoordArray(0, tx);

    quad->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));
    state = quad->getOrCreateStateSet();
    state->setTextureAttributeAndModes(0, texture.get());

    state = quad->getOrCreateStateSet();
    state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    osg::Geode *geode = new osg::Geode;
    geode->addDrawable(quad);
    return geode;
}
