#include <cmath>
#include <numeric>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/local_parameterization.h>
#include <mlib/ceres_util/triangulation.h>
#include <mlib/ceres_util/costfunctions.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/random.h>

#include <mlib/sfm/solvers/essential_matrix_hartley_gpl/ematrix_hartley_gpl.h>
#include <mlib/sfm/solvers/essential_matrix_nghiaho_bsd/5point.h>
#include <mlib/sfm/solvers/geometry_tools.h>
#include <mlib/sfm/solvers/essential_matrix_solver.h>
#include <mlib/ceres_util/unit_length_cost.h>

using namespace std;
namespace cvl{

//// 5-point solver wrappers //////////////////////////////////////

std::vector<Matrix3d> computeEssentialMatrices_Hartley(const std::vector<Vector2d>& pts1, const std::vector<Vector2d>& pts2)
{
	using namespace hartley;

	vector<Matrix3d> candidates;

	int nroots;
	Ematrix Ematrices[10];

	Matches_5 q1, q2;

	for (int i = 0; i < 5; i++) {
		q1[i][0] = pts1[i](0);
		q1[i][1] = pts1[i](1);
		q1[i][2] = 1;
		q2[i][0] = pts2[i](0);
		q2[i][1] = pts2[i](1);
		q2[i][2] = 1;
	}

	compute_E_matrices(q1, q2, Ematrices, nroots, true);
	candidates.resize(nroots);

	for (int k = 0; k < nroots; k++) {
		for (int j = 0; j < 3; j++) {
			for (int i = 0; i < 3; i++) {
				candidates[k](i, j) = Ematrices[k][i][j];
			}
		}
	}

	return candidates;
}

std::vector<Matrix3d> computeEssentialMatrices_NghiaHo(const std::vector<Vector2d>& pts1, const std::vector<Vector2d>& pts2)
{
	using namespace nghiaho;

	vector<nghiaho::EMatrix> E; // essential matrix
	vector<nghiaho::PMatrix> P; // 3x4 projection matrix
	vector<int> inliers;

	vector<Matrix3d> candidates;

	// This solver can use N points but is restricted to the minimal case (N=5) to be comparable with Hartleys implementation.
	vector<double> q1(10), q2(10);

	for (int i = 0; i < 5; i++) {
		q1[2 * i] = pts1[i](0);
		q1[2 * i + 1] = pts1[i](1);
		q2[2 * i] = pts2[i](0);
		q2[2 * i + 1] = pts2[i](1);
	}

	Solve5PointEssential(&q1[0], &q2[0], 5, E, P, inliers);
	candidates.resize(E.size());

	for (size_t k = 0; k < E.size(); k++) {
		for (int j = 0; j < 3; j++) {
			for (int i = 0; i < 3; i++) {
				candidates[k](i, j) = E[k](i, j);
			}
		}
	}

	return candidates;
}

//// Common solver methods ///////////////////////////////////

std::set<uint> EssentialMatrixSolverBase::getUniqueRandomInts(int minval, int maxval, size_t n)
{
	std::set<uint> v;

	while (v.size() < n) {
        v.insert(mlib::randu<double>(minval, maxval));
	}

	return v;
}

uint EssentialMatrixSolverBase::RANSACUpdateNumIters(double p, double outlier_ratio, uint model_points, uint max_iters)
{

	p = std::max(std::min(p, 1.0), 0.0);
	outlier_ratio = std::max(std::min(outlier_ratio, 1.0), 0.0);
	double dbl_min = std::numeric_limits<double>::min();

	// avoid inf's & nan's
	double num = std::max(1.0 - p, dbl_min);
	double denom = 1.0 - pow(1.0 - outlier_ratio, model_points);
	if (denom < dbl_min)
		return 0;

	num = log(num);
	denom = log(denom);

	return (denom >= 0 || -num >= max_iters*(-denom)) ? max_iters : (uint)floor(num / denom + 0.5);
}

//// 2-view essential matrix solver //////////////////////////

EssentialMatrixSolver::EssentialMatrixSolver()
{
	ematrixMinSolve = computeEssentialMatrices_Hartley; // Hartley's own implementation

	// Finds fewer candidates than Hartley's implementation but seems to produce more consistent results
	// ematrixMinSolve = computeEssentialMatrices_NghiaHo;

}

void EssentialMatrixSolver::setParameters(const Parameters& params)
{
	this->parameters = params;
	is_configured = true;
}

bool isNumericallySound(const Matrix3d& E)
{
	double Esum = 0;
	double Emax = 0;

	for (uint i = 0; i < 9; i++) {
        double a = fabs(E[i]);
		Emax = (a > Emax) ? a : Emax;
		Esum += a;
	}

	if (Esum < 1E-10) return false;
	Esum = Esum / Emax; // Eigen: Ecand.cwiseAbs().sum() / Ecand.lpNorm<Infinity>()
	return !std::isnan(Esum) && !std::isinf(Esum);
}

bool isValidScale(double num, double den)
{
	float eps = 1E-3;

	num = fabs(num);
	den = fabs(den);

	if (num < eps || den < eps) return false;
	float s = num / den;
	if (std::isinf(s) || std::isnan(s)) return false;

	return true;
}

void getDisjointRandomIndexSets(uint n1, uint n2, uint N, std::set<uint>& set1, std::set<uint>& set2)
{
	while (set1.size() < n1) {
        set1.insert(mlib::randui<int>(0, N - 1));
	}

	while (set2.size() < n2) {
        int i = mlib::randui<int>(0, N - 1);
		if (set1.find(i) != set1.end()) continue; // Skip points in set1
		set2.insert(i);
	}
}

#if 0
// Deprecated, will be removed

uint EssentialMatrixSolver::estimateFromMinimal(const std::vector<Vector2d>& y1n_pts,
    const std::vector<Vector2d>& y2n_pts, Matrix3d *E) const
{
	assert(is_configured);
	const Parameters& p = parameters;
	assert(y1n_pts.size() == y2n_pts.size());

	const uint num_points = y1n_pts.size();
	assert(num_points >= 5 + p.minimal_test_sz);

	// Set up minimal-estimation and verification point-pair index sets

	set<uint> estimation_set; // Index set of points used to estimate E
	set<uint> test_set; // Index set of points used to select the correct E candidate

	while (estimation_set.size() < 5) {
        estimation_set.insert(mlib::randu<double>(0, num_points - 1));
	}

	while (test_set.size() < p.minimal_test_sz){
        int i = mlib::randu<double>(0, num_points - 1);
		if (estimation_set.find(i) != estimation_set.end()) continue; // Avoid points in the estimation set.
		test_set.insert(i);
	}

	// Estimate and select the best E candidate
	// The test is used to find the E candidate with the most inliers

	vector<Matrix3d> candidates = ematrixMinSolve(
		selectFromIndexSet(y1n_pts, estimation_set), selectFromIndexSet(y2n_pts, estimation_set));

	set<uint> cset;	// Consensus index set
	uint good_candidates = 0;

	if (candidates.size() == 0) {
		goto done;
	}

	for (auto& Ecand : candidates) {

		// Ensure the minimal solver actually returned a good candidate:
		// the matrix should be nonzero, and have no infs or nans
		// all points in the estimation set should be on the epipolar lines

		if (!isNumericallySound(Ecand)) continue;

		bool bad_solution = false;
		for (uint i : estimation_set) {
			double e = fabs(getEpipolarLineDistance(Ecand, y1n_pts[i], y2n_pts[i]));
			if (e > 1E-14) {
				bad_solution = true;
				break;
			}
		}
		if (bad_solution) continue;

		// Find the test-set inliers, using epipolar-line distance error
		// store the E matrix if the result is better than for previous candidates

		set<uint> inlier_set;

		for (uint i : test_set) {
			double e = fabs(getEpipolarLineDistance(Ecand, y1n_pts[i], y2n_pts[i]));
			if (e < p.ransac_max_d) {
				inlier_set.insert(i);
			}
		}

		if (inlier_set.size() > cset.size()) {
			cset = inlier_set;
			*E = Ecand;
			good_candidates++;
		}
	}

	if (cset.size() < p.minimal_test_sz * 0.5) { // Require at least 50% inliers
		good_candidates = 0;
		goto done;
	}

done:
	return good_candidates;

}


bool EssentialMatrixSolver::estimate(const std::vector<Vector2d>& y1n_pts,
    const std::vector<Vector2d>& y2n_pts, PoseD *P,
    std::vector<bool>& inlier_flags, uint& num_inliers, uint& num_iters, PoseD *P_notrefined) const
{
	assert(is_configured);
	assert(y1n_pts.size() == y2n_pts.size());

	const Parameters& p = parameters;
	const uint model_points = 5; // Number points needed for the minimal case
	const uint max_failures = 100*p.ransac_max_iters; // Number of retries before giving up.
	const uint num_points = y1n_pts.size();

	num_inliers = 0;

	if (num_points < model_points + p.minimal_test_sz) {
		return false; // No E matrix can be estimated
	}

	inlier_flags.resize(num_points, false);

	// Estimate P (RANSAC loop + optimization)

	uint max_iters = p.ransac_max_iters; // Current maximum number of iterations the RANSAC loop will perform
	float maxd = p.ransac_max_d; // Epipolar line distance threshold
	uint num_failures = 0; // Number of times minimal solver returned without any hypothesis
	num_iters = 0; // Number of iterations made

	for (uint i = 0; i < max_iters + num_failures && num_failures < max_failures; i++)
	{
		Matrix3d E;
		
		// Estimate an E and find the epipolar-line inliers

		uint num_hypotheses = estimateFromMinimal(y1n_pts, y2n_pts, &E);
		if (num_hypotheses == 0) {
			num_failures++;
			continue;
		}
		/*int inlier_cnt = */getEpipolarLineInliers(maxd, y1n_pts, y2n_pts, E, inlier_flags);

		// Get four camera candidates and pick the one with most reprojection inliers

        vector<PoseD> cameras = extractCamerasFromE(E);

		vector<bool> best_iflags;
		uint best_cam_ix = 0;
		uint best_icnt = 0;
		for (uint j = 0; j < 4; j++) {
			auto& cam = cameras[j];
			vector<bool> iflags = inlier_flags; // Copy
			uint icnt = getReprojectionInliers(p.max_reproj_error, y1n_pts, y2n_pts, cam, iflags);
			if (icnt > best_icnt) {
				best_icnt = icnt;
				best_iflags = iflags;
				best_cam_ix = j;
			}
		}

		inlier_flags = best_iflags;

		// Store the result if it is the best so far

		if (best_icnt > num_inliers) {
			*P = cameras[best_cam_ix];
			num_inliers = best_icnt;

			// Update the estimate of the maximum number of iterations
			double error_p = (double)(num_points - num_inliers) / num_points; // error probability
			max_iters = RANSACUpdateNumIters(p.ransac_answer_confidence, error_p, model_points, max_iters);
		}
		num_iters++;

		if (num_inliers > num_points * p.early_success_iratio) {
			break; // Quit early if the inlier ratio is good enough
		}
	}

	fprintf(stderr, "E-matrix: %d inliers, %d failures, %d iters\n", num_inliers, num_failures, num_iters);

	if (num_inliers == 0) {
		return false;
	}

	if (P_notrefined) {
		*P_notrefined = *P;
	}

    refine(y1n_pts, y2n_pts, &inlier_flags, P);
    refine(y1n_pts, y2n_pts, &inlier_flags, P);
	return true;
}

#endif

uint EssentialMatrixSolver::estimateMinimalAndTest(
    const std::vector<Vector2d>& est_y1s, const std::vector<Vector2d>& est_y2s,
    const std::vector<Vector2d>& test_y1s, const std::vector<Vector2d>& test_y2s,
    Matrix3d& E, vector<bool>& test_inliers, uint& num_test_inliers) const
{
	assert(is_configured);
	const Parameters& p = parameters;

	uint good_candidates = 0;
	test_inliers = vector<bool>(test_y1s.size(), false);
	num_test_inliers = 0;

	// Estimate and select the best E candidate
	// The test is used to find the E candidate with the most inliers

	vector<Matrix3d> candidates = ematrixMinSolve(est_y1s, est_y2s);

	if (candidates.size() == 0) {
		goto done;
	}

	for (auto& E_candidate : candidates) {

		// Ensure the minimal solver actually returned a good candidate:
		// the matrix should be nonzero, and have no infs or nans
		// all points in the estimation set should be on the epipolar lines

		if (!isNumericallySound(E_candidate)) continue;

		bool bad_solution = false;
		uint n = (uint)est_y1s.size();

		for (uint i = 0; i < n; i++) {
			double e = fabs(getEpipolarLineDistance(E_candidate, est_y1s[i], est_y2s[i]));
			if (e > 1E-14) {
				bad_solution = true;
				break;
			}
		}
		if (bad_solution) continue;

		// Find the test-set inliers, using epipolar-line distance error
		// store the E matrix if the result is better than for previous candidates

		n = (uint)test_y1s.size();
		vector<bool> iflags(n);
		uint icnt = 0;

		for (uint i = 0; i < n; i++) {
			double e = fabs(getEpipolarLineDistance(E_candidate, test_y1s[i], test_y2s[i]));
			iflags[i] = (e < p.ransac_max_d);
			if (iflags[i]) {
				icnt++;
			}
		}

		if (icnt > num_test_inliers) {
			test_inliers = iflags;
			num_test_inliers = icnt;
			E = E_candidate;
			good_candidates++;
		}
	}

	if (num_test_inliers < p.minimal_test_sz * 0.5) { // Require at least 50% inliers
		good_candidates = 0;
		goto done;
	}

done:
	return good_candidates;
}


uint EssentialMatrixSolver::getCamera(const std::vector<Vector2d>& y1s, const std::vector<Vector2d>& y2s,
    const Matrix3d& E, std::vector<bool>& inliers, PoseD& P) const
{
	assert(is_configured);
	const Parameters& p = parameters;

	// Get four camera candidates and pick the one with most reprojection inliers

    vector<PoseD> cameras = extractCamerasFromE(E);

	vector<bool> best_iflags;
	uint best_cam_ix = 0;
	uint best_icnt = 0;
	for (uint j = 0; j < 4; j++) {
		auto& cam = cameras[j];
		vector<bool> iflags = inliers;
		uint icnt = getReprojectionInliers(p.max_reproj_error, y1s, y2s, cam, iflags);
		if (icnt > best_icnt) {
			best_icnt = icnt;
			best_iflags = iflags;
			best_cam_ix = j;
		}
	}
	P = cameras[best_cam_ix];
	inliers = best_iflags;

	if (best_icnt == 0) {
		return 0;
	}

	// Optimize the result 

	refine(y1s, y2s, &inliers, &P);

	// Count inliers after refinement
	uint icnt = 0;
	for (auto i : inliers) {
		icnt += i ? 1 : 0;
	}

	return icnt;
}

bool EssentialMatrixSolver::estimate(const std::vector<Vector2d>& y1s,
    const std::vector<Vector2d>& y2s,
    PoseD& P,
	std::vector<bool>& inliers,
	uint& num_inliers,
	uint& num_iters) const
{
	assert(is_configured);
	assert(y1s.size() == y2s.size());
	uint num_points = (uint)y1s.size();

	const Parameters& p = parameters;
	const uint model_points = 5; // Number points needed for the minimal case
	const uint max_failures = 100*p.ransac_max_iters; // Number of retries before giving up.


	if (num_points < model_points + p.minimal_test_sz) {
		return false; // No E matrix can be estimated
	}

	inliers = vector<bool>(num_points, false);
	num_inliers = 0;

	// Estimate P (RANSAC loop + optimization)

	uint max_iters = p.ransac_max_iters; // Current maximum number of iterations the RANSAC loop will perform
	uint num_failures = 0; // Number of times minimal solver returned without any hypothesis
	num_iters = 0; // Number of iterations made

	for (uint i = 0; i < max_iters + num_failures && num_failures < max_failures; i++)
	{
		Matrix3d E_candidate;
        PoseD P_candidate;

		// Estimate an E and find its epipolar-line inliers

		std::set<uint> estimation_set, test_set;
		vector<Vector2d> est_y1s, est_y2s, test_y1s, test_y2s;
		vector<bool> test_inliers;
		uint num_test_inliers;

		getDisjointRandomIndexSets(5, p.minimal_test_sz, num_points, estimation_set, test_set);

		est_y1s = selectFromIndexSet(y1s, estimation_set);
		est_y2s = selectFromIndexSet(y2s, estimation_set);

		test_y1s = selectFromIndexSet(y1s, test_set);
		test_y2s = selectFromIndexSet(y2s, test_set);

		uint num_hypotheses = estimateMinimalAndTest(est_y1s, est_y2s, test_y1s, test_y2s, E_candidate, test_inliers, num_test_inliers);
		if (num_hypotheses == 0) {
			num_failures++;
			continue;
		}

		// Get a camera from the E, with the test set and
		// test the reprojection error against all point correspondences

		num_test_inliers = getCamera(test_y1s, test_y2s, E_candidate, test_inliers, P_candidate);
		vector<bool> iflags(num_points, true);
		uint icnt = getReprojectionInliers(p.max_reproj_error, y1s, y2s, P_candidate, iflags);

		// Store the result if it is the best so far

		if (icnt > num_inliers) {
			P = P_candidate;
			num_inliers = icnt;
			inliers = iflags;

			// Update the estimate of the maximum number of iterations
			double error_p = (double)(num_points - num_inliers) / num_points; // error probability
			max_iters = RANSACUpdateNumIters(p.ransac_answer_confidence, error_p, model_points, max_iters);
		}
		num_iters++;

		if (num_inliers > num_points * p.early_success_iratio) {
			fprintf(stderr, "Have good inlier ratio - stopping early, %d inliers of %d correspondences (%.2f%%)\n", num_inliers, num_points,
				num_inliers * 100.0f / num_points);
			break; // Quit early if the inlier ratio is good enough
		}
	}

	fprintf(stderr, "E-matrix: %d inliers, %d failures, %d iters\n", num_inliers, num_failures, num_iters);

	if (num_inliers == 0) {
		return false;
	}

	refine(y1s, y2s, &inliers, &P);
	refine(y1s, y2s, &inliers, &P);
	return true;
}

struct PointReprojectionError
{
	Vector2d y1, y2;

	PointReprojectionError(const Vector2d& y1, const Vector2d& y2)
		: y1(y1), y2(y2)
	{
	}

	template <typename T>
	bool operator()(const T* const quaternion, const T* const translation, const T* const X, T* residuals) const
	{

		// Reproject X into the identity camera

		T yr1[3];

		yr1[0] = X[0] / X[2];
		yr1[1] = X[1] / X[2];

		// Reproject X into the camera to optimize over

		T yr2[3];
		ceres::UnitQuaternionRotatePoint(quaternion, X, yr2);
		yr2[0] += translation[0];
		yr2[1] += translation[1];
		yr2[2] += translation[2];

		yr2[0] = yr2[0] / yr2[2];
		yr2[1] = yr2[1] / yr2[2];

		// Get reprojection errors

		residuals[0] = yr1[0] - y1(0);
		residuals[1] = yr1[1] - y1(1);
		residuals[2] = yr2[0] - y2(0);
		residuals[3] = yr2[1] - y2(1);

		return true;
	}

	static ceres::CostFunction* Create(const Vector2d& y1, const Vector2d& y2)
	{
		return (new ceres::AutoDiffCostFunction<PointReprojectionError, 4, 4, 3, 3>(
			new PointReprojectionError(y1, y2)));
	}
};

void EssentialMatrixSolver::refine(const std::vector<Vector2d>& y1n_pts, const std::vector<Vector2d>& y2n_pts,
    std::vector<bool> *inliers, PoseD *P) const
{
	using namespace mlib;

	assert(is_configured);
	assert(y1n_pts.size() == y2n_pts.size());
	const Parameters& p = parameters;

	// Assume P1 = [I|0]
    PoseD& P2 = *P;

	size_t n = y1n_pts.size();
	assert(y2n_pts.size() == n);

	// Triangulate inliers 

    MidpointTriangulator mpt(PoseD(), P2);
	vector<Vector3d> triangulations;
	vector<Vector2d> observations1, observations2;

	triangulations.reserve(n);
	observations1.reserve(n);
	observations2.reserve(n);

	for (size_t i = 0; i < n; i++) {
        //if (!(*inliers)[i]) continue;

		// Triangulate

		const Vector2d& y1 = y1n_pts[i];
		const Vector2d& y2 = y2n_pts[i];

		Vector3d X = mpt.triangulate(y1, y2);


		// Validate

		if (std::isnan(X(0)) || std::isnan(X(1)) || std::isnan(X(2)) ||
            std::isinf(X(0)) || std::isinf(X(1)) || std::isinf(X(2))||(X(2)<0))
		{
			(*inliers)[i] = false;
			continue;
		}

		double e1 = (y1 - X.hnormalized()).length();
		double e2 = (y2 - (P2 * X).hnormalized()).length();

		if (e1 + e2 > 2 * p.max_reproj_error) {
			(*inliers)[i] = false;
			continue;
		}

		// Store

		observations1.push_back(y1n_pts[i]);
		observations2.push_back(y2n_pts[i]);
		triangulations.push_back(X);
		(*inliers)[i] = true;
	}
	n = triangulations.size();

	// Prepare solver

	ceres::Problem problem;
	ceres::Solver::Options options;

	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = false;

    options.max_num_iterations = 15;
	options.function_tolerance = 1e-8;     // default 1e-6
	options.gradient_tolerance = 1e-10;     //default 1e-4*function_tolerance

    //ceres::LossFunction* loss = nullptr;
    ceres::LossFunction* loss = new ceres::HuberLoss(p.max_reproj_error);

	// P1 is [I|0]
	double *P2_q = P2.getRRef();
	double *P2_t = P2.getTRef();

	for (size_t i = 0; i < n; i++) {

		Vector2d& y1 = observations1[i];
		Vector2d& y2 = observations2[i];
		Vector3d& X = triangulations[i];

		ceres::CostFunction* cost_fn = PointReprojectionError::Create(y1, y2);
		problem.AddResidualBlock(cost_fn, loss, P2_q, P2_t, &X(0));
	}
    problem.AddResidualBlock(UnitLengthError<3>::Create(),nullptr,P2_t);

	ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering();
	ordering->AddElementToGroup(P2_q, 1);
	ordering->AddElementToGroup(P2_t, 2);
	for (size_t i = 0; i < n; i++) {
		Vector3d& X = triangulations[i];
		ordering->AddElementToGroup(&X(0), 0);
    }

    options.linear_solver_ordering.reset(ordering);

	problem.SetParameterization(P2_q, new ceres::QuaternionParameterization());

	// Optimize

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
//	cout << summary.FullReport() << endl;

	// P2 already has the output since Ceres modifies it directly.
}

//// Essential matrix scale solver ///////////////////////////



EssentialMatrixScaleSolver::EssentialMatrixScaleSolver(const PoseD& Pa1, const PoseD& Pa2, const PoseD& Pb1)
	: Pa1(Pa1), Pa2(Pa2), Pb1(Pb1)
{
	Tba = Pb1 * Pa1.inverse(); // Known relative transform from camera a to b
	Ta21 = Pa2 * Pa1.inverse(); // Relative transform from Pa1 to Pa2

	auto Rba = Tba.rotation(); // Must not be identity
	auto tba = Tba.translation();

	auto Tab = Tba.inverse();
	auto Rab = Tab.rotation();
	auto tab = Tab.translation();

	auto Ra21 = Ta21.rotation();
	auto ta21 = Ta21.translation();

	auto Rb21 = Rba * Ra21 * Rab; // Relative rotation from Pb1 to Pb2

	A = (tba + Rba * Ra21 * tab).crossMatrix() * Rb21;
	B = (Rba * ta21).crossMatrix() * Rb21;

	//int id = 1;
	//printf("[%d] Tba = %s\n", id, toString(Tba).c_str());
	//printf("[%d] Ta21 = %s\n", id, toString(Ta21).c_str());
	//printf("[%d] Tab = %s\n", id, toString(Tab).c_str());
	//printf("[%d] Rab = %s\n", id, toString(Ra21).c_str());
	//printf("[%d] Ra21 = %s\n", id, toString(Ra21).c_str());
	//printf("[%d] Rb21 = %s\n", id, toString(Rb21).c_str());
	//printf("[%d] A = %s\n", id, toString(A).c_str());
	//printf("[%d] B = %s\n", id, toString(B).c_str());
}

void EssentialMatrixScaleSolver::setParameters(const Parameters& params){
	this->parameters = params;
	is_configured = true;
}

bool EssentialMatrixScaleSolver::estimateMinimal(const Vector2d& yb1, const Vector2d& yb2,
	double& scale_numerator, double& scale_denominator) const
{
	auto y1 = yb1.homogeneous();
	auto y2 = yb2.homogeneous();
	scale_numerator = y2.dot(A * y1);
	scale_denominator = (-y2).dot(B * y1);

    float eps = 1E-6;

	float num = fabs(scale_numerator);
	float den = fabs(scale_denominator);
	float s = num / den;

	return (num > eps && den > eps && !std::isinf(s) && !std::isnan(s));
}

void EssentialMatrixScaleSolver::estimateAll(const PoseD& Pa1, const PoseD& Pa2,
    const PoseD& Pb1, const std::vector<Vector2d>& yb1s,
    const std::vector<Vector2d>& yb2s, std::vector<double>& scale)
{
    PoseD Tba = Pb1 * Pa1.inverse(); // Known relative transform from camera a to b
    PoseD Ta21 = Pa2 * Pa1.inverse(); // Relative transform from Pa1 to Pa2

	auto Rba = Tba.rotation(); // Must not be identity
	auto tba = Tba.translation();

	auto Tab = Tba.inverse();
	auto Rab = Tab.rotation();
	auto tab = Tab.translation();

	auto Ra21 = Ta21.rotation();
	auto ta21 = Ta21.translation();

	auto Rb21 = Rba * Ra21 * Rab; // Relative rotation from Pb1 to Pb2

	auto A = (tba + Rba * Ra21 * tab).crossMatrix() * Rb21;
	auto B = (Rba * ta21).crossMatrix() * Rb21;

	//int id = 2;
	//printf("[%d] Tba = %s\n", id, toString(Tba).c_str());
	//printf("[%d] Ta21 = %s\n", id, toString(Ta21).c_str());
	//printf("[%d] Tab = %s\n", id, toString(Tab).c_str());
	//printf("[%d] Rab = %s\n", id, toString(Ra21).c_str());
	//printf("[%d] Ra21 = %s\n", id, toString(Ra21).c_str());
	//printf("[%d] Rb21 = %s\n", id, toString(Rb21).c_str());
	//printf("[%d] A = %s\n", id, toString(A).c_str());
	//printf("[%d] B = %s\n", id, toString(B).c_str());

	uint n = (uint)yb1s.size();

	scale.clear();
	scale.reserve(n);

	for (uint i = 0; i < n; i++) {
		auto y1 = yb1s[i].homogeneous();
		auto y2 = yb2s[i].homogeneous();

		double s_num = y2.dot(A * y1);
		double s_den = (-y2).dot(B * y1);

		scale.push_back(s_num / s_den);
	}
}

bool EssentialMatrixScaleSolver::estimate(const std::vector<Vector2d>& yb1s,
    const std::vector<Vector2d>& yb2s, double& scale, uint& num_inliers, uint& num_iters)
{
	assert(is_configured);
	const Parameters& p = parameters;
	assert(yb1s.size() == yb2s.size());

	const uint num_points = yb1s.size();
	assert(num_points >= 1 + p.minimal_test_sz);

	const uint model_points = 1; // Number points needed for the minimal case
	const uint max_failures = p.ransac_max_iters; // Number of retries before giving up.

	num_inliers = 0;

	if (num_points < model_points + p.minimal_test_sz) {
		return false; // The scale cannot be estimated
	}


    // Count inliers for the scale =1.0
    double minratio;
    {
        PoseD Pb2 = Tba * PoseD(Pa2.rotation(), Pa2.translation());
        PoseD Pb21 = Pb2 * Pb1.inverse(); // Relative pose

        vector<bool> inliers(yb1s.size(), true);
        uint icnt = getReprojectionInliers(p.max_reproj_error,
            yb1s, yb2s, Pb21, inliers);
        minratio=(double)icnt/(double)yb1s.size();
    }
    //the scale inlier count must surpass the minratio!

	// Estimate P (RANSAC loop + optimization)

	uint max_iters = p.ransac_max_iters; // Current maximum number of iterations the RANSAC loop will perform
	uint num_failures = 0; // Number of times minimal solver returned without any hypothesis
	num_iters = 0; // Number of iterations made

	for (uint i = 0; i < max_iters + num_failures && num_failures < max_failures; i++)
	{
		// Select estimation and test point pairs

        uint est_ix = mlib::randui<int>(0, num_points - 1); // Index of pair to estimate scale from
		set<uint> test_set; // Index set of points used to verify the correct scale

		while (test_set.size() < p.minimal_test_sz){
            uint i = mlib::randui<int>(0, num_points - 1);
			if (i == est_ix) continue; // Avoid The point used for the estimate
			test_set.insert(i);
		}

		// Estimate the scale

		double s_num, s_den;
		bool success = estimateMinimal(yb1s[est_ix], yb2s[est_ix], s_num, s_den);
		if (!success) {
			num_failures++;
			continue;
		}
		double s = s_num / s_den;

		bool update_num_iters = false;

		// Count inliers in the test set (positive scale)
		{
            PoseD Pb2 = Tba * PoseD(Pa2.rotation(), s * Pa2.translation());
            PoseD Pb21 = Pb2 * Pb1.inverse(); // Relative PoseD

			vector<bool> inliers(test_set.size(), true);
			uint icnt = getReprojectionInliers(p.max_reproj_error,
				selectFromIndexSet(yb1s, test_set), selectFromIndexSet(yb2s, test_set), Pb21, inliers);

			if (icnt > num_inliers) {
				scale = s;
				num_inliers = icnt;
				update_num_iters = true;
			}
		}
/*
        // Count inliers in the test set (negative scale) Shouldnt be nec, the direction of the translation has been tested!
		{
			s = -s;

            PoseD Pb2 = Tba * PoseD(Pa2.rotation(), s * Pa2.translation());
            PoseD Pb21 = Pb2 * Pb1.inverse(); // Relative PoseD

			vector<bool> inliers(test_set.size(), true);
			uint icnt = getReprojectionInliers(p.max_reproj_error,
				selectFromIndexSet(yb1s, test_set), selectFromIndexSet(yb2s, test_set), Pb21, inliers);

			if (icnt > num_inliers) {
				scale = s;
				num_inliers = icnt;
				update_num_iters = true;
			}
		}
*/

		// Update the estimate of the maximum number of iterations

		if (update_num_iters) {
			double error_p = (double)(test_set.size() - num_inliers) / test_set.size(); // error probability
			max_iters = RANSACUpdateNumIters(p.ransac_answer_confidence, error_p, model_points, max_iters);
		}


		// Count inliers in the test set (negative scale)
		num_iters++;

		if (num_inliers > num_points * p.early_success_iratio) {
			break; // Quit early if the inlier ratio is good enough
		}
    }
    if(num_inliers<minratio*p.minimal_test_sz)
        scale=1.0;
    num_inliers=minratio*p.minimal_test_sz;

//	fprintf(stderr, "Scale: s = %f, %d inliers, %d failures, %d iters\n", scale, num_inliers, num_failures, num_iters);

	return (num_inliers > 0);
}

ESPoseEstimator::ESPoseEstimator(const float flen_a, const float flen_b, const float flen_c)
{
	es_params = getDefaultParamsE(flen_a);
	ss_params = getDefaultParamsScale(flen_b);
	this->flen_c = flen_c;
}

EssentialMatrixSolver::Parameters ESPoseEstimator::getDefaultParamsE(float focal_length)
{
	EssentialMatrixSolver::Parameters p;

	p.ransac_answer_confidence = 0.9995;
	p.ransac_max_d = 2 / focal_length;
	p.max_reproj_error = 2 / focal_length;
	p.ransac_max_iters = 1000;
	p.early_success_iratio = 0.85;
	p.minimal_test_sz = 50;

	return p;
}

EssentialMatrixScaleSolver::Parameters ESPoseEstimator::getDefaultParamsScale(float focal_length)
{
	EssentialMatrixScaleSolver::Parameters p;

	p.ransac_answer_confidence = 0.9995;
	p.max_reproj_error = 2 / focal_length;
	p.ransac_max_iters = 1000;
	p.early_success_iratio = 0.85;
	p.minimal_test_sz = 50;

	return p;
}

PoseD ESPoseEstimator::estimatePose(const PoseD& Tba, const PoseD& Tca,
    const std::vector<Vector2d>& ya1s, const std::vector<Vector2d>& ya2s,
    const std::vector<Vector2d>& yb1s, const std::vector<Vector2d>& yb2s,
    const std::vector<Vector2d>& yc1s, const std::vector<Vector2d>& yc2s,
	std::vector<std::vector<bool>>& inliers, uint& num_inliers, uint& num_iters)
{
	// Outline:
	// 1. Call EssentialMatrixSolver to estimate Ea from ya1s, ya2s (has built-in RANSAC)
	// 2. RANSAC:
	//    Estimate scale from Ea + small set of points in yb1s, yb2s
	//    Recover Pa2 and compute Pb2, (optionally Pc2)
	//    Check reprojection errors in all cameras
	//    Keep new Pa2 if better than before
	// 3. Return Pa2

    PoseD Pa1; // [I|0]
    PoseD Pb1 = Tba * Pa1;
    PoseD Pc1 = Tca * Pa1;

	// Estimate Ea from points seen by camera A.

	EssentialMatrixSolver ems;
	ems.setParameters(es_params);

	assert(ya1s.size() == ya2s.size());

	uint ninliers, niters;

	vector<bool> inliers_a, inliers_b, inliers_c;
	vector<bool> best_inliers_b, best_inliers_c;

    PoseD Pa21;
	ems.estimate(ya1s, ya2s, Pa21, inliers_a, ninliers, niters);
    PoseD Pa2_0 = Pa21 * Pa1; // PoseD with unit translation

	// Estimate the scale

	EssentialMatrixScaleSolver scale_solver(Pa1, Pa2_0, Tba);
	scale_solver.setParameters(ss_params);

	uint max_iters = es_params.ransac_max_iters; // Current maximum number of iterations the RANSAC loop will perform
	uint max_failures = max_iters;
	uint model_points = 1; // FIXME: 1 or ss_params.minimal_test_sz?
	uint num_failures = 0; // Number of times minimal solver returned without any hypothesis
	num_iters = 0; // Number of iterations made

	double scale = 1.0;

	for (uint i = 0; i < max_iters + num_failures && num_failures < max_failures; i++)
	{
		// Estimate the scale and new camera Pa2 from Ea and points seen by camera B

        double s=1.0;
		bool success = scale_solver.estimate(yb1s, yb2s, s, ninliers, niters);

        if (!success) {
        //	num_failures++;
        //	continue;
            s=1.0;
        }
        PoseD Pa2 = Tba * PoseD(Pa2_0.rotation(), s * Pa2_0.translation());


		// Count inliers in camera B

		uint icnt_b = 0, icnt_c = 0;
//		bool update_num_iters = false;

		if (!yb1s.empty()) {

            PoseD Pb2 = Tba * Pa2;
            PoseD Pb21 = Pb2 * Pb1.inverse();

			vector<bool> inliers(yb1s.size(), true);
			icnt_b = getReprojectionInliers(es_params.max_reproj_error, yb1s, yb2s, Pb21, inliers_b);
		}

		// Count inliers in camera C

		if (!yc1s.empty()) {

            PoseD Pc2 = Tca * Pa2;
            PoseD Pc21 = Pc2 * Pc1.inverse();

			vector<bool> inliers(yc1s.size(), true);
			icnt_c = getReprojectionInliers(es_params.max_reproj_error, yc1s, yc2s, Pc21, inliers_c);
		}

		if (icnt_b + icnt_c > num_inliers) {
			scale = s;
			num_inliers = icnt_b + icnt_c;

			best_inliers_b = inliers_b;
			best_inliers_c = inliers_c;

			uint set_sz = yc1s.size();

			double error_p = (double)(set_sz - num_inliers) / set_sz; // error probability
			max_iters = RANSACUpdateNumIters(es_params.ransac_answer_confidence, error_p, model_points, max_iters);

//			update_num_iters = true;
		}
	}

	inliers.push_back(inliers_a);
	inliers.push_back(inliers_b);
	inliers.push_back(inliers_c);

    /// \todo: Refine the scale by combining all scale estimates

    return PoseD(Pa2_0.rotation(), scale * Pa2_0.translation());

}
}
