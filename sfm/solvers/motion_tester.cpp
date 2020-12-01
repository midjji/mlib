#include <mlib/utils/vector.h>
#include <mlib/sfm/solvers/motion_tester.h>

using namespace cvl;

bool MotionTester::haveMotion()
{
	if (n < parameters.min_num_points) return false;

	computeDisparities();
	return testDisparities();
}

bool MotionTester::haveTranslation(const PoseD& P21)
{
	if (n < parameters.min_num_points) return false;

	computeDisparities(&P21);
	return testDisparities();
}

void MotionTester::computeDisparities(const cvl::PoseD* P21 /*= nullptr*/)
{
	if (P21 == nullptr) {
		for (uint i = 0; i < n; i++) {
			mvectors[i] = y1s[i] - y2s[i];
		}
	}
	else {
		// Compensate for camera rotation so that translational motion can be observed
        PoseD R = PoseD(P21->getQuaternion(), Vector3d(0, 0, 0));
		for (uint i = 0; i < n; i++) {
			mvectors[i] = (R * y1s[i].homogeneous()).hnormalized() - y2s[i];
		}
	}

	for (uint i = 0; i < n; i++) {
		disparities[i] = parameters.focal_length * mvectors[i].length();
	}
}

bool MotionTester::testDisparities()
{
	const Parameters& p = parameters;

	float mu = mlib::mean(disparities);
    float sigma = sqrt(mlib::variance(disparities, mu));

	float small_disp_thr = p.min_disparity_sigma * p.corner_noise_stddev;

	uint num_survivors = 0;
	for (uint i = 0; i < n; i++) {
		if (disparities[i] > p.outlier_sigma * sigma) continue; // Discard probable outliers
		if (disparities[i] < small_disp_thr) continue; // Discard small disparities
		num_survivors++;
	}

	if (num_survivors < n * p.test1_survivor_ratio) return false;
	return true;
}

MotionTester::MotionTester(const std::vector<cvl::Vector2d>& y1s, const std::vector<cvl::Vector2d>& y2s)
	: y1s(y1s), y2s(y2s), n(y1s.size()), mvectors(n), disparities(n)
{
	assert(y2s.size() == n);
}
