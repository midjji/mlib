
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/MatrixTransform>
#include <osg/Point>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/StateSetManipulator>
#include <osg/LineWidth>
#include <cassert>

#include "mlib/vis/GLTools.h"
#include <mlib/vis/CvGL.h>
#include <mlib/vis/convertosg.h>

osg::ref_ptr<osg::Node> MakePointCloud(osg::ref_ptr<osg::Vec3Array> vertices,
                                       osg::ref_ptr<osg::Vec3Array> colors,
                                       float ptSize)
{
	assert(vertices->size() == colors->size());
    osg::ref_ptr<osg::Geometry> geo = new osg::Geometry;
	geo->setVertexArray(vertices);
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
	normals->push_back(osg::Vec3(0, 0, 0)); // Dummy value (necessary when using BIND_OFF?)
	geo->setNormalArray(normals, osg::Array::BIND_OFF);
	geo->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
	geo->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, vertices->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
	geode->addDrawable(geo);

	osg::PolygonMode *pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::POINT);
    osg::Point *psz = new osg::Point(ptSize);

	osg::StateSet *state = geode->getOrCreateStateSet();
	state->setAttributeAndModes(psz, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
	state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    //state->setMode(GL_POINT_SMOOTH, osg::StateAttribute::ON);
    //state->setMode(GL_BLEND, osg::StateAttribute::ON);


    osg::Group* group=new osg::Group();
    group->addChild(geode);
    return group;
}

osg::Geode *MakeImagePlane(osg::Image *img)
{
	osg::Geode *imgPlane = new osg::Geode;

    float hw = float(img->s() / 2);
    float hh = float(img->t() / 2);
    float f = -1.0f;

	osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
	vertices->push_back(osg::Vec3(-hw, -hh, f));
	vertices->push_back(osg::Vec3(hw, -hh, f));
	vertices->push_back(osg::Vec3(hw, hh, f));
	vertices->push_back(osg::Vec3(-hw, hh, f));

	osg::ref_ptr<osg::Vec3Array> normal = new osg::Vec3Array;
	normal->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));

	osg::ref_ptr<osg::Vec2Array> texCoords = new osg::Vec2Array;
	texCoords->push_back(osg::Vec2(0, 1));
	texCoords->push_back(osg::Vec2(1, 1));
	texCoords->push_back(osg::Vec2(1, 0));
	texCoords->push_back(osg::Vec2(0, 0));

	osg::Vec3Array *color = new osg::Vec3Array;
	color->push_back(osg::Vec3(1, 1, 1));

	osg::ref_ptr<osg::Geometry> quad = new osg::Geometry;
	quad->setVertexArray(vertices);
	quad->setColorArray(color, osg::Array::BIND_PER_PRIMITIVE_SET);
	quad->setNormalArray(normal, osg::Array::BIND_OVERALL);
	quad->setTexCoordArray(0, texCoords);
	quad->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));
	imgPlane->addDrawable(quad);

	osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
	texture->setImage(0, img);
	texture->setResizeNonPowerOfTwoHint(false);

	osg::StateSet *state = imgPlane->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0, texture);
	state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);

	return imgPlane;
}

osg::MatrixTransform *MakeCameraIcon(osg::ref_ptr<osg::Image> photo, int width, int height,
                                    const osg::Matrix& pose, float flen, float f, const osg::Vec3& color)
{
    osg::StateSet *state;

    // Camera image

    if (photo.valid()) {
        width = photo->s();
        height = photo->t();
    }

    float hw = 0.5f * f * (float(width) / flen); // image plane half width
    float hh = 0.5f * f * (float(height) / flen); // image plane half height

    osg::Geode *img_plane = nullptr;

    // Camera image plane

    if (photo != nullptr) {
        img_plane = new osg::Geode;

        osg::ref_ptr<osg::Vec3Array> img_vertices = new osg::Vec3Array;
        img_vertices->push_back(osg::Vec3(-hw, -hh, f));
        img_vertices->push_back(osg::Vec3( hw, -hh, f));
        img_vertices->push_back(osg::Vec3( hw,  hh, f));
        img_vertices->push_back(osg::Vec3(-hw,  hh, f));
        img_vertices->push_back(osg::Vec3(0,0,0));

        osg::ref_ptr<osg::Vec3Array> img_normal = new osg::Vec3Array;
        img_normal->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));

        osg::ref_ptr<osg::Vec2Array> img_texcoords = new osg::Vec2Array;
        img_texcoords->push_back(osg::Vec2(0, 1));
        img_texcoords->push_back(osg::Vec2(1, 1));
        img_texcoords->push_back(osg::Vec2(1, 0));
        img_texcoords->push_back(osg::Vec2(0, 0));

        osg::Vec3Array *img_color = new osg::Vec3Array;
        img_color->push_back(osg::Vec3(1,1,1));

        osg::ref_ptr<osg::Geometry> img_quad = new osg::Geometry;
        img_quad->setVertexArray(img_vertices);
        img_quad->setColorArray(img_color, osg::Array::BIND_PER_PRIMITIVE_SET);
        img_quad->setNormalArray(img_normal, osg::Array::BIND_OVERALL);
        img_quad->setTexCoordArray(0, img_texcoords);
        img_quad->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

        osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
        texture->setImage(0, photo);

        img_plane->addDrawable(img_quad);

        state = img_plane->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0, texture);
        state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    }

    // Camera frustum pyramid

    osg::Vec3Array *cam_vertices = new osg::Vec3Array;
    cam_vertices->push_back(osg::Vec3(0,0,0));
    cam_vertices->push_back(osg::Vec3(-hw, -hh, f));
    cam_vertices->push_back(osg::Vec3( hw, -hh, f));
    cam_vertices->push_back(osg::Vec3( hw,  hh, f));
    cam_vertices->push_back(osg::Vec3(-hw,  hh, f));
    cam_vertices->push_back(osg::Vec3(-hw, -hh, f));

    osg::Vec3Array *cam_color = new osg::Vec3Array;
    cam_color->push_back(color);

    osg::Geometry *cam_pyramid = new osg::Geometry;
    cam_pyramid->setVertexArray(cam_vertices);
    cam_pyramid->setColorArray(cam_color, osg::Array::BIND_PER_PRIMITIVE_SET);
    cam_pyramid->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN, 0, 6));

    osg::Geode *cam_frustum = new osg::Geode;
    cam_frustum->addDrawable(cam_pyramid);

    osg::PolygonMode *pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);

    state = cam_frustum->getOrCreateStateSet();
    state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    state->setMode(GL_LINE_SMOOTH, osg::StateAttribute::ON);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);

    // Camera up vector

    osg::Vec3Array *upv_vertices = new osg::Vec3Array;
    upv_vertices->push_back(osg::Vec3(0,0,0));
    upv_vertices->push_back(osg::Vec3(0, -hh/2, 0));

    osg::Vec3Array *upv_color = new osg::Vec3Array;
    upv_color->push_back(osg::Vec3(0,1,0));

    osg::Geometry *upv_line = new osg::Geometry;
    upv_line->setVertexArray(upv_vertices);
    upv_line->setColorArray(upv_color, osg::Array::BIND_PER_PRIMITIVE_SET);
    upv_line->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 2));

    osg::Geode *cam_upvector = new osg::Geode;
    cam_upvector->addDrawable(upv_line);

    state = cam_upvector->getOrCreateStateSet();
    state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    state->setMode(GL_LINE_SMOOTH, osg::StateAttribute::ON);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);

    osg::MatrixTransform *model = new osg::MatrixTransform;

	model->setMatrix(pose);

    if (img_plane != nullptr) {
        model->addChild(img_plane);
    }
    model->addChild(cam_frustum);
    model->addChild(cam_upvector);

    return model;
}

osg::Node *MakeAxisMarker(float axis_length, float line_width, const osg::Matrix& pose)
{
    osg::Vec3Array *vertices = new osg::Vec3Array;
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(axis_length,0,0));
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(0,axis_length,0));
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(0,0,axis_length));

    osg::Vec3Array *colors = new osg::Vec3Array;
    colors->push_back(osg::Vec3(1,0,0));
    colors->push_back(osg::Vec3(0,1,0));
    colors->push_back(osg::Vec3(0,0,1));

    osg::Geometry *lines = new osg::Geometry;
    lines->setVertexArray(vertices);
    lines->setColorArray(colors, osg::Array::BIND_PER_PRIMITIVE_SET);
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 2));
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 2, 2));
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 4, 2));

    osg::Geode *marker_geode = new osg::Geode;
    marker_geode->addDrawable(lines);

    osg::PolygonMode *pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    osg::LineWidth *lw = new osg::LineWidth(line_width);

    osg::StateSet *state = marker_geode->getOrCreateStateSet();
    state->setAttributeAndModes(lw, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    state->setMode(GL_LINE_SMOOTH, osg::StateAttribute::ON);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);

    osg::MatrixTransform *marker_tform = new osg::MatrixTransform;
    marker_tform->setMatrix(pose);
    marker_tform->addChild(marker_geode);

    return marker_tform;
}


osg::Geode *MakeAxisMarker(float axis_length, float line_width)
{
    osg::Vec3Array *vertices = new osg::Vec3Array;
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(axis_length,0,0));
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(0,axis_length,0));
    vertices->push_back(osg::Vec3(0,0,0));
    vertices->push_back(osg::Vec3(0,0,axis_length));

    osg::Vec3Array *colors = new osg::Vec3Array;
    colors->push_back(osg::Vec3(1,0,0));
    colors->push_back(osg::Vec3(0,1,0));
    colors->push_back(osg::Vec3(0,0,1));

    osg::Geometry *lines = new osg::Geometry;
    lines->setVertexArray(vertices);
    lines->setColorArray(colors, osg::Array::BIND_PER_PRIMITIVE_SET);
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 2));
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 2, 2));
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 4, 2));

    osg::Geode *marker = new osg::Geode;
    marker->addDrawable(lines);

    osg::PolygonMode *pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    osg::LineWidth *lw = new osg::LineWidth(line_width);

    osg::StateSet *state = marker->getOrCreateStateSet();
    state->setAttributeAndModes(lw, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    state->setMode(GL_LINE_SMOOTH, osg::StateAttribute::OFF);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);

    return marker;
}

osg::MatrixTransform *MakeAxisMarker(const osg::Matrixd& pose, float axis_length, float line_width)
{
    osg::Geode *marker0 = MakeAxisMarker(axis_length, line_width);
    osg::MatrixTransform *marker = new osg::MatrixTransform;
    marker->setMatrix(pose);
    marker->addChild(marker0);
    return marker;
}








osg::Group* MakeTrajectory(const std::vector<osg::Matrixd>& poses, float length,float width)
{
    osg::Group* group=new osg::Group();
    for(const auto& pose:poses)
        group->addChild(MakeAxisMarker(length,width,pose));
    return group;
}

osg::Node *MakeGrid(int num_squares, float side_length, const osg::Vec3& color, const osg::Matrix& pose)
{
    int i;
    float x;
    float z = 0.0f;
    float L = (side_length * num_squares) / 2;

    osg::Vec3Array *vertices = new osg::Vec3Array;

    // Create two sets of serpentine (winding) lines, to form a grid.

    // Add lines parallel to the X-axis

    for (i = 0, x = -L; i < num_squares + 1; i++, x += side_length)
    {
        float dir = (i % 2 == 0) ? 1 : -1;
        vertices->push_back(osg::Vec3(-dir*L, x, z));
        vertices->push_back(osg::Vec3( dir*L, x, z));
    }

    vertices->push_back(osg::Vec3(L, -L, z));

    // Add lines parallel to the Y-axis

    for (i = 0, x = -L; i < num_squares + 1; i++, x += side_length)
    {
        float dir = (i % 2 == 0) ? 1 : -1;
        vertices->push_back(osg::Vec3(x, -dir*L, z));
        vertices->push_back(osg::Vec3(x,  dir*L, z));
    }

    osg::Vec3Array *colors = new osg::Vec3Array;
    colors->push_back(color);

    osg::Geometry *lines = new osg::Geometry;
    lines->setVertexArray(vertices);
    lines->setColorArray(colors, osg::Array::BIND_OVERALL);
    lines->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, vertices->size()));

    osg::Geode *grid = new osg::Geode;
    grid->addDrawable(lines);

    osg::PolygonMode *pm = new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    osg::LineWidth *lw = new osg::LineWidth(1.0);

    osg::StateSet *state = grid->getOrCreateStateSet();
    state->setAttributeAndModes(lw, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setAttributeAndModes(pm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    state->setMode(GL_LINE_SMOOTH, osg::StateAttribute::ON);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);

    osg::MatrixTransform *transform = new osg::MatrixTransform;
    transform->setMatrix(pose);
    transform->addChild(grid);

    return transform;
}

osgViewer::View *ConfigureOsgWindow(osgViewer::View *view, const std::string& name, int x, int y, unsigned int width, unsigned int height)
{
	osg::GraphicsContext::WindowingSystemInterface* wsi = osg::GraphicsContext::getWindowingSystemInterface();
	if (!wsi) {
		osg::notify(osg::NOTICE) << "Error, no WindowSystemInterface available, cannot create windows." << std::endl;
        return nullptr;
	}
	unsigned int scrWidth, scrHeight;
	wsi->getScreenResolution(osg::GraphicsContext::ScreenIdentifier(0), scrWidth, scrHeight);

	osg::ref_ptr<osg::Camera> camera = view->getCamera();

	view->addEventHandler(new osgGA::StateSetManipulator(camera->getOrCreateStateSet()));
	//view->addEventHandler(new osgViewer::StatsHandler());
	//view->addEventHandler(new osgViewer::WindowSizeHandler());

	osg::GraphicsContext::Traits *traits = new osg::GraphicsContext::Traits;
	traits->x = x;
	traits->y = y;
	traits->width = std::min(scrWidth, (unsigned)width);
	traits->height = std::min(scrHeight, (unsigned)height);
	traits->windowDecoration = true;
	traits->doubleBuffer = true;
    traits->sharedContext = nullptr;
	traits->windowName = name;

	osg::ref_ptr<osg::GraphicsContext> gc = osg::GraphicsContext::createGraphicsContext(traits);
	if (gc.valid()) {
		osg::notify(osg::INFO) << "  GraphicsWindow has been created successfully." << std::endl;
		gc->setClearColor(osg::Vec4f(0.2f, 0.2f, 0.6f, 1.0f));
		gc->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}
	else {
		osg::notify(osg::NOTICE) << "  GraphicsWindow has not been created successfully." << std::endl;
        return nullptr;
	}

	view->setName(name); // The window, view and graphics context have the same name for convenience.
	gc->setName(name); // The graphics context will also have the same name

	camera->setGraphicsContext(gc);
	camera->setViewport(0, 0, traits->width, traits->height);

	return view;
}

osgViewer::View *MakeOsgWindow(const std::string& name, int x, int y, unsigned int width, unsigned int height)
{
	return ConfigureOsgWindow(new osgViewer::View, name, x, y, width, height);
}

void ConfigureOrthoCamera(int width, int height, osg::Camera* camera)
{
	camera->setProjectionMatrix(osg::Matrix::ortho2D(-width / 2, width / 2, -height / 2, height / 2));
	camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
	camera->setViewMatrix(osg::Matrix::identity());
	camera->setClearMask(GL_DEPTH_BUFFER_BIT);
	camera->setAllowEventFocus(true);
}

