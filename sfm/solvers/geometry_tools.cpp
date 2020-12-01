#include <eigen3/Eigen/SVD>
#include <mlib/sfm/solvers/geometry_tools.h>
#include <mlib/ceres_util/triangulation.h>

using namespace std;

namespace cvl{
//// Essential matrix tools ////////////////////////////////////////////////

double getEpipolarLineDistance(const Matrix3d& E,
    const Vector2d& x1, const Vector2d& x2)
{
	Vector3d L = E * x1.homogeneous();
    double Z = 1.0/(std::sqrt(L(0)*L(0) + L(1)*L(1)));

    return x2.homogeneous().dot(L*Z);
}

void decomposeEssentialMatrix(const Matrix3d& E, Matrix3d& U, Vector3d& S, Matrix3d& V)
{
	typedef Eigen::Matrix<double, 3, 3> Mat3;
	typedef Eigen::Matrix<double, 3, 1> Vec3;
	typedef Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> RowMajorMap;

	// Normalize E

    Mat3 E2 = RowMajorMap(const_cast<double*>(E.data())); // Copy row-major matrix to Eigen
	Eigen::JacobiSVD<Mat3> svd(E2, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Vec3 s = svd.singularValues();
	S = Vector3d(s(0), s(1), s(2));
    RowMajorMap(U.data()) = svd.matrixU();
    RowMajorMap(V.data()) = svd.matrixV();
}

Matrix3d normalizeEssentialMatrix(const Matrix3d& E)
{
	typedef Eigen::Matrix<double, 3, 3> Mat3;
	typedef Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> RowMajorMap;

    Matrix3d Eout;

    Mat3 E_ = RowMajorMap(const_cast<double*>(E.data())); // Copy row-major matrix to Eigen
	Eigen::JacobiSVD<Mat3> svd(E_, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Mat3 S;
	S <<
		1, 0, 0,
		0, 1, 0,
		0, 0, 0;

    RowMajorMap(Eout.data()) = svd.matrixU() * S * (svd.matrixV().transpose());

	return Eout;
}

vector<PoseD> extractCamerasFromE(const Matrix3d& E)
{
    vector<PoseD> cameras;

	// Reference: H&Z pp. 258-260

    PoseD P1 = PoseD(); // First camera: [I|0] transformation
    PoseD P2; // Second camera: [R|t] - sought

	typedef Eigen::Matrix<double, 3, 3> Mat3;
	typedef Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> RowMajorMap;

	// Decompose E

	Matrix3d U, V;

    Mat3 E_ = RowMajorMap(const_cast<double*>(E.data()));
	Eigen::JacobiSVD<Mat3> svd(E_, Eigen::ComputeFullU | Eigen::ComputeFullV);
     /// \todo: Verify that E is a matrix, s.t s(0) == s(1), s(2) = 0 (?)
    //Vec3 s = svd.singularValues();
    RowMajorMap(U.data()) = svd.matrixU();
    RowMajorMap(V.data()) = svd.matrixV();

	// Generate possible solutions

	Matrix3d W(
		0, -1, 0,
		1, 0, 0,
		0, 0, 1);

	Matrix3d R1 = U * W * V.transpose();
	R1 = R1 * R1.determinant(); // Ensure det(R1) = 1
	Matrix3d R2 = U * W.transpose() * V.transpose();
	R2 = R2 * R2.determinant();
    Vector3d u3 = U.Col(2);

    cameras.push_back(PoseD(R1, u3));
    cameras.push_back(PoseD(R1, -u3));
    cameras.push_back(PoseD(R2, u3));
    cameras.push_back(PoseD(R2, -u3));

	return cameras;
}

uint getEpipolarLineInliers(double max_d, const std::vector<Vector2d>& y1s, const std::vector<Vector2d>& y2s,
    const Matrix3d& E, std::vector<bool>& inliers)
{
	const uint num_points = y1s.size();
	assert(y2s.size() == num_points);
	inliers.resize(num_points);

	uint inlier_cnt = 0;
	for (uint j = 0; j < num_points; j++) {
		double d = fabs(getEpipolarLineDistance(E, y1s[j], y2s[j]));
		inliers[j] = (d < max_d);
		if (inliers[j]) {
			inlier_cnt++;
		}
	}

	return inlier_cnt;
}

uint getReprojectionInliers(double max_e, const std::vector<Vector2d>& y1s, const std::vector<Vector2d>& y2s,
    const PoseD& P2, std::vector<bool>& inliers)
{
	const uint num_points = y1s.size();
	assert(y2s.size() == num_points);
	inliers.resize(num_points);

    MidpointTriangulator mpt(PoseD(), P2);

	uint inlier_cnt = 0;
	uint i1 = 0, i2 = 0, i3 = 0;
	for (uint i = 0; i < num_points; i++) {

		if (!inliers[i] ) continue;

		inliers[i] = false;

		Vector2d y1 = y1s[i];
		Vector2d y2 = y2s[i];

		Vector3d X = mpt.triangulate(y1, y2);

		if (std::isnan(X(0)) || std::isnan(X(1)) || std::isnan(X(2)) ||
			std::isinf(X(0)) || std::isinf(X(1)) || std::isinf(X(2)))
		{
			inliers[i] = false;
			continue;
		}

		i1++;

		// Reproject and check check camera P1 = [I|0]

		Vector3d x1 = X; // x1 = [I|0] * X

		if (x1(2) < 0) {
			continue; // Behind camera
		}

		i2++;

		if ((y1 - x1.hnormalized()).length() > max_e) {
			continue;
		}

		// Reproject and check camera P2

		Vector3d x2 = P2 * X;

		if (x2(2) < 0) {
			continue; // Behind camera
		}

		if ((y2 - x2.hnormalized()).length() > max_e) {
			continue;
		}

		inliers[i] = true;
		inlier_cnt++;
		i3++;
	}

	//fprintf(stderr, "i1=%d i2=%d i3=%d --\n", i1, i2, i3);

	return inlier_cnt;
}


int selectCameraFromE(const std::vector<PoseD>& cameras, const Vector2d& y1, const Vector2d& y2)
{
    PoseD P1;

	for (int i = 0; i < 4; i++) {
        const PoseD& P2 = cameras[i];

        Vector3d X = triangulate(P1, P2, y1, y2);
		Vector3d x1 = P1 * X;
        if (x1[2] > 0) {
			Vector3d x2 = P2 * X;
            if (x2[2] > 0) {
				return i;
			}
		}
	}
	return -1;
}

uint selectCameraFromE(double max_e, const std::vector<PoseD>& cameras,
    const std::vector<Vector2d>& y1s, const std::vector<Vector2d>& y2s,
	std::vector<bool>& inliers, uint& inlier_count)
{
	vector<bool> best_iflags(y1s.size(), false);
	uint best_cam_ix = 0;
	uint best_icnt = 0;
	for (uint j = 0; j < 4; j++) {
		vector<bool> iflags = inliers; // Copy
		uint icnt = getReprojectionInliers(max_e, y1s, y2s, cameras[j], iflags);
		if (icnt > best_icnt) {
			best_icnt = icnt;
			best_iflags = iflags;
			best_cam_ix = j;
		}
	}

	if (best_iflags.empty()) {
		// Found no inliers whatsoever.
		inliers = vector<bool>(y1s.size(), false);
	}
	else {
		inliers = best_iflags;
	}

	inlier_count = best_icnt;
	return best_cam_ix;
}

PoseD extractCameraFromE(const Matrix3d& E, const Vector2d& y1n, const Vector2d& y2n)
{
	// Reference: H&Z pp. 258-260

    PoseD P1 = PoseD(); // First camera: [I|0] transformation
    PoseD P2; // Second camera: [R|t] - sought

	typedef Eigen::Matrix<double, 3, 3> Mat3;
	typedef Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> RowMajorMap;

	// Decompose E

	Matrix3d U, V;
	
	Mat3 E_;
	//E_ = RowMajorMap(const_cast<double*>(&E.a00));
    for(int i=0;i<9;++i)E_(i)=E(i);


	Eigen::JacobiSVD<Mat3> svd(E_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //Vec3 s = svd.singularValues();
    /// \todo: Verify that E is a matrix, s.t s(0) == s(1), s(2) = 0 (?)

    RowMajorMap(U.data()) = svd.matrixU();
    RowMajorMap(V.data()) = svd.matrixV();

	// Generate possible solutions

	Matrix3d W(
		0, -1, 0,
		1, 0, 0,
		0, 0, 1);

	Matrix3d R1 = U * W * V.transpose();
	R1 = R1 * R1.determinant(); // Ensure det(R1) = 1
	Matrix3d R2 = U * W.transpose() * V.transpose();
	R2 = R2 * R2.determinant();
    Vector3d u3 = U.Col(2);

	// Test and select one of the solutions

	for (int i = 0; i < 4; i++) {
		switch (i) {
        case 0: P2 = PoseD(R1, u3); break;
        case 1: P2 = PoseD(R1, -u3); break;
        case 2: P2 = PoseD(R2, u3); break;
        case 3: P2 = PoseD(R2, -u3); break;
            default:;

		}

        Vector3d X = triangulate(P1, P2, y1n, y2n);
		Vector3d x1 = P1 * X;
        if (x1[2] > 0) {
			Vector3d x2 = P2 * X;
            if (x2[2] > 0) {
				return P2;
			}
		}
	}

	return P2; // Never reached (?)
}

//// 2-view midpoint triangulation /////////////////////////////////////////

MidpointTriangulator::MidpointTriangulator(const PoseD& P1, const PoseD& P2, double d)
	: P1(P1),
	pose1(P1.inverse()),
	c1(pose1.translation()),
	P2(P2),
	pose2(P2.inverse()),
	c2(pose2.translation()),
	far_distance(d)
{
}

Vector3d MidpointTriangulator::triangulate(const Vector2d& y1, const Vector2d& y2) const
{
	Vector3d X;
	Vector3d pa, pb;
	double mua, mub;

	// Create lines in the world frame, from the camera centers to the observed 2D points y1,y2

	Vector3d y1c1 = pose1 * y1.homogeneous(); // point from camera 1, moved into the world frame
	Vector3d y2c2 = pose2 * y2.homogeneous(); // point from camera 2, moved into the world frame

	if (!lineLineIntersect(c1, y1c1, c2, y2c2, pa, pb, mua, mub)) {
		// No midpoint found, place the 3D point "far away", in front of a camera.
		X = pose2 * Vector3d(0, 0, far_distance);
	}
	else {
		X = 0.5 * (pa + pb);
	}

	Vector3d x1 = P1 * X;
	Vector3d x2 = P2 * X;

    if (x1[2] < 0 && x2[2] < 0) {
		// The 3D point is behind both cameras, probably due to
		// uncertainty in the observations of a far-away image feature.
		// Fix this by placing the point in front of one camera.
        x1[2] = -x1[2];
		X = pose1 * x1;
	}

	return X;
}

bool MidpointTriangulator::lineLineIntersect(const Vector3d& p1, const Vector3d& p2,
    const Vector3d& p3, const Vector3d& p4, Vector3d& pa, Vector3d& pb, double& mua, double& mub) const
{
	Vector3d p13, p43, p21;
	double d1343, d4321, d1321, d4343, d2121;
	double numer, denom;

	const double eps = std::numeric_limits<double>::epsilon();

	p13 = p1 - p3;

	p43 = p4 - p3;
    if (fabs(p43[0]) + fabs(p43[1]) + fabs(p43[2]) < 3 * eps) {
		return false;
	}

	p21 = p2 - p1;
    if (fabs(p21[0]) + fabs(p21[1]) + fabs(p21[2]) < 3 * eps) {
		return false;
	}

	d1343 = p13.dot(p43);
	d4321 = p43.dot(p21);
	d1321 = p13.dot(p21);
	d4343 = p43.dot(p43);
	d2121 = p21.dot(p21);

	denom = d2121 * d4343 - d4321 * d4321;
	if (fabs(denom) < eps) {
		return false;
	}

	numer = d1343 * d4321 - d1321 * d4343;

	mua = numer / denom;
	mub = (d1343 + d4321 * mua) / d4343;

	pa = p1 + mua * p21;
	pb = p3 + mub * p43;

	return true;
}
}
