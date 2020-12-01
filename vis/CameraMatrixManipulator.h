#pragma once
#include <osgGA/CameraManipulator>

/// A simple camera manipulator for controlling cameras with camera matrices (i.e inverse of world pose).
class CameraMatrixManipulator : public osgGA::CameraManipulator
{
	osg::Matrixd position;

public:
	CameraMatrixManipulator()
	{
		position = osg::Matrixd::identity();
	}

	/** Set the position of the matrix manipulator using a 4x4 GL Matrix.*/
	virtual void setByMatrix(const osg::Matrixd& matrix)
	{
		position = matrix;
	}

	/** Set the position of the matrix manipulator using a 4x4 GL Matrix.*/
	virtual void setByInverseMatrix(const osg::Matrixd& matrix)
	{
		position = osg::Matrixd::inverse(matrix);
	}

	/** Get the position of the manipulator as 4x4 GL Matrix.*/
	virtual osg::Matrixd getMatrix() const
	{
		return position;
	}

	/** Get the position of the manipulator as a inverse GL matrix of the manipulator, typically used as a model view matrix.*/
	virtual osg::Matrixd getInverseMatrix() const
	{
		return osg::Matrixd::inverse(position);
	}
};