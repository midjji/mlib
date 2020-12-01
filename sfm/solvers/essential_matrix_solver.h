#pragma once
#include <set>
#include <vector>
#include <mlib/utils/cvl/pose.h>
namespace cvl{




class EssentialMatrixSolverBase
{
public:
	/// Get n unique integer values in the range [minval, maxval]
	static std::set<uint> getUniqueRandomInts(int minval, int maxval, size_t n);

	/** Extract vector elements selected by an index set
	*
	* @param data    Vector to extract elements from.
	* @pram indices  Indices of elements to extract.
	*
	* @returns Extracted elements
	*/
	template <typename T>
	static std::vector<T> selectFromIndexSet(const std::vector<T>& data, const std::set<uint>& indices)
	{
		std::vector<T> selected(indices.size());

		int j = 0;
		for (auto& i : indices) {
			selected[j] = data[i];
			j++;
		}

		return selected;
	}

	/** Update the maximum number of iterations a RANSAC solver may perform, based on current results.
	*
	* @param p              RANSAC success probability, e.g 0.99
	* @param outlier_ratio  Percentage of outliers in the current RANSAC iteration.
	* @param model_points   Number of data points needed by the minimal solver inside the RANSAC loop
	* @param max_iters      Current maximum number of iterations the RANSAC solver is allowed to do.
	*
	* @returns New maximum number of iterations.
	*
	* @note  This function is ported from the C-API of OpenCV, where it is named cvRANSACUpdateNumIters().
	*/
	static uint RANSACUpdateNumIters(double p, double outlier_ratio, uint model_points, uint max_iters);
};

/**
 * A basic essential-matrix estimator. The E matrix is estimated with a 5-point solver [1] in
 * a standard RANSAC loop followed by nonlinear optimization of the inlier set.
 *
 * [1] Li, Hongdong, and Richard Hartley. "Five-point motion estimation made easy."
 * Pattern Recognition, 2006. ICPR 2006. 18th International Conference on.
 * Vol. 1. IEEE, 2006.
 *
 * Some definitions to help remembering what is what:
 *
 * X = 3D space vector
 * P1 = [I | 0] = normalized camera projection matrix, first camera
 * P2 = [R | t] = normalized camera projection matrix, second camera
 *
 * With x1 = P1*X, x2 = P2*X and x2^T E x1 = 0 then E = Cross{t} * R
 * where Cross{t} = cross-product matrix of relative camera translation
 * and          R = relative camera rotation
 */
class EssentialMatrixSolver : public EssentialMatrixSolverBase
{
public:

    typedef std::vector<Matrix3d> (*MinimalSolverFn)(const std::vector<Vector2d>&, const std::vector<Vector2d>&);
	
	struct Parameters
	{
		uint minimal_test_sz;			//!< Size of the test set used to select the minimal E solution

		float ransac_max_d;				//!< RANSAC inlier maximum epipolar-line distance (K-normalized)
		float max_reproj_error;			//!< Inlier reprojection error threshold (K-normalized pixels)
		uint ransac_max_iters;			//!< RANSAC maximum number of iterations allowed
		float ransac_answer_confidence;	//!< RANSAC confidence in answer (0.0 - 1.0), try 0.995
		float ransac_failure_iratio;	//!< RANSAC signals complete failure if the inlier ratio is lower
		float early_success_iratio;		//!< RANSAC early-exit inlier ratio: stop if the current inlier ratio is higher.
	};

	Parameters parameters;
	bool is_configured;

	EssentialMatrixSolver();

	//!< Selected minimal solver function
	MinimalSolverFn ematrixMinSolve;

	/** Configure the solver */
	void setParameters(const Parameters& params);

#if 0
	// Deprecated, will be removed

	/** Estimate E in the minimal (5-point) case.
	*
	* @param y1n_pts  K-normalized 2D points from camera 1
	* @param y2n_pts  K-normalized 2D points from camera 2 of the same size as y1n_pts
	* @param E        [Output] The essential matrix
	*
	* @returns  Zero on failure, or the number of good candidates that were evaluated.
	*/
    uint estimateFromMinimal(const std::vector<Vector2d>& y1n_pts,
        const std::vector<Vector2d>& y2n_pts, Matrix3d *E) const;

	/** Estimate E from N points
	*
	* @param y1n_pts      K-normalized 2D points from camera 1
	* @param y2n_pts      K-normalized 2D points from camera 2
	* @param P            [Output] Camera projection [R|t]. E is computed as E = crossMatrix(t) * R.
	* @param inlier_flags [Output] Vector of same length as input points, marking inliers 1, outliers 0.
	* @param num_inliers  [Output] Number inliers with the returned E.
	* @param num_iters    [Output] Number of iterations in the RANSAC loop
	*
	* @returns  True on success
	*/
    bool estimate(const std::vector<Vector2d>& y1n_pts, const std::vector<Vector2d>& y2n_pts,
        PoseD *P, std::vector<bool>& inlier_flags, uint& num_inliers, uint& num_iters, PoseD *P_notrefined = nullptr) const;
#endif

	/** Estimate E in the minimal (5-point) case and test it against some correspondences.
	*
	* @param est_y1s          K-normalized 2D points from camera 1, used for estimating E. Must be of size 5.
	* @param est_y2s          K-normalized 2D points from camera 2 of the same size as est_y1s
	* 
	* @param test_y1s         K-normalized 2D points from camera 1, used for testing E. Must not share any points with est_y1s.
	* @param test_y2s         K-normalized 2D points from camera 2 of the same size as test_y1s
	*
	* @param E                [Output] The essential matrix
	* @param test_inliers     [Output] Vector of the same size as test_y1s, marking test inliers true, outliers false.
	* @param num_test_inliers [Output] Number of inliers in the test set
	*
	* @returns  Zero on failure, or the number of good candidates that were evaluated.
	*/
	uint estimateMinimalAndTest(
        const std::vector<Vector2d>& est_y1s, const std::vector<Vector2d>& est_y2s,
        const std::vector<Vector2d>& test_y1s, const std::vector<Vector2d>& test_y2s,
        Matrix3d& E, std::vector<bool>& test_inliers, uint& num_test_inliers) const;

	/**
	 * Extract the relative camera transform from an essential matrix.
	 * The method selects the transform has the most inlier support
	 *
	 * @param y1s      K-normalized 2D points from camera 1
	 * @param y2s      K-normalized 2D points from camera 2
	 * @param E        Essential matrix
	 *
	 * @param inliers  [Input/output] y1s/y2s inlier flags.
	 *                 Expects inliers to be used for camera extraction while ignoring outliers.
	 *                 Returns the reprojection inliers of the extracted camera.
	 *
	 * @param P        [Output] Extracted camera transform
	 *
	 * @returns inlier count
	 */
    uint getCamera(const std::vector<Vector2d>& y1s, const std::vector<Vector2d>& y2s,
        const Matrix3d& E, std::vector<bool>& inliers, PoseD& P) const;

	/** Estimate E from N points
	*
	* @param y1s          K-normalized 2D points from camera 1
	* @param y2s          K-normalized 2D points from camera 2
	* @param P            [Output] Camera projection [R|t]. E is computed as E = crossMatrix(t) * R.
	* @param inliers      [Output] Vector of same length as input points, marking inliers 1, outliers 0.
	* @param num_inliers  [Output] Number inliers with the returned E.
	* @param num_iters    [Output] Number of iterations in the RANSAC loop
	*
	* @returns  True on success
	*/
    bool estimate(const std::vector<Vector2d>& y1s, const std::vector<Vector2d>& y2s,
        PoseD& P, std::vector<bool>& inliers, uint& num_inliers, uint& num_iters) const;
	
	/** Refine the estimated E with all inlier point pairs.
	*
	* @param y1n_pts  K-normalized 2D points from camera 1
	* @param y2n_pts  K-normalized 2D points from camera 2
	* @param inliers  [Input/Output] Vector of same length as the input points, flagging inliers true, outliers false.
	*
	* @param P2       [Input/Output] The camera projection [R|t] to work on. This is related to E = Cross{t} * R.
	*/
    void refine(const std::vector<Vector2d>& y1n_pts, const std::vector<Vector2d>& y2n_pts,
        std::vector<bool> *inliers, PoseD *P) const;
};

/**
* Compute candidate E matrices using Hartley's 5 point solver implementation.
*
* @param pt1 5 normalized image 2D points from the first camera.
* @param pt2 5 normalized image 2D points from the second camera.
*
* @return A vector of up to 10 3x3 E-matrix candidates satisfying the condition x2^T*E*x1 = 0
*         for all points x1 in pt1 and x2 in pt2.
*/
std::vector<Matrix3d> computeEssentialMatrices_Hartley(const std::vector<Vector2d>& pts1, const std::vector<Vector2d>& pts2);

/**
* Compute candidate E matrices using Nghia Ho's 5 point solver implementation.
*
* @param pt1 5 normalized image 2D points from the first camera.
* @param pt2 5 normalized image 2D points from the second camera.
*
* @return A vector of up to 10 3x3 E-matrix candidates satisfying the condition x2^T*E*x1 = 0
*         for all points x1 in pt1 and x2 in pt2.
*/
std::vector<Matrix3d> computeEssentialMatrices_NghiaHo(const std::vector<Vector2d>& pts1, const std::vector<Vector2d>& pts2);

struct EssentialMatrixScaleSolver : public EssentialMatrixSolverBase
{
    PoseD Pa1;	//!< Camera transform of the primary camera at time 1
    PoseD Pa2;	//!< Camera transform of the primary camera at time 2
    PoseD Ta21;	//!< Relative transform of the primary camera from time t = 1 to t = 2, i.e Pa2 * Pa1.inverse();

    PoseD Pb1;	//!< Camera transform of the secondary camera at time 2
    PoseD Tba;	//!< Known relative transform from camera a to b, i.e Pa2 * Pa1.inverse()

    Matrix3d A, B;

	struct Parameters
	{
		uint minimal_test_sz;			//!< Size of the test set used to verify the minimal scale solution

// Not used:		float ransac_max_d;				//!< RANSAC inlier maximum epipolar-line distance (K-normalized)
		float max_reproj_error;			//!< Inlier reprojection error threshold (K-normalized pixels)
		uint ransac_max_iters;			//!< RANSAC maximum number of iterations allowed
		float ransac_answer_confidence;	//!< RANSAC confidence in answer (0.0 - 1.0), try 0.995
		float ransac_failure_iratio;	//!< RANSAC signals complete failure if the inlier ratio is lower
		float early_success_iratio;		//!< RANSAC early-exit inlier ratio: stop if the current inlier ratio is higher.
	};

	Parameters parameters;
	bool is_configured;

	/** Constructor.
	 *
	 * @param Pa1  Camera transform of the primary camera at time 1
	 * @param Pa2  Camera transform of the primary camera at time 2
	 *
	 * @param Pb1  Camera transform of the secondary camera at time 2.
	 * This camera is assumed to have a fixed relative pose to the primary camera.
	 */
    EssentialMatrixScaleSolver(const PoseD& Pa1, const PoseD& Pa2, const PoseD& Pb1);

	/** Configure the solver */
	void setParameters(const Parameters& params);

    static void estimateAll(const PoseD& Pa1, const PoseD& Pa2, const PoseD& Pb1,
        const std::vector<Vector2d>& yb1s, const std::vector<Vector2d>& yb2s, std::vector<double>& scale);

	/** Estimate the scale using one pair of observations from the secondary camera
	 *
	 * @param yb1  K-normalized 2D observation at at time t = 1
	 * @param yb2  K-normalized 2D observation at at time t = 2
	 *
	 * @param scale_numerator    [Output] Scale estimate numerators for each of the 2D points. Divide by the
	 *                           denominators to obtain the scales. Watch out for degenerate cases where this
	 *                           division is numerically ill posed, e.g the denominators are zero.
	 * @param scale_denominator  [Output] Scale estimate denominators.
	 *
	 * @return true on success, false if the estimate is bad.
	 *
	 * @note It is important that the primary (Pa) and secondary (Pb) cameras are rotated in relation
	 * to one another, i.e that the camera image planes are not coplanar. This can checked by
	 * verifying that the rotation component of Tba is not identity.
	 *
	 * Furthermore, the motion of the cameras from instance 1 to 2 must also have a rotation,
	 * that is, the rotation component of Pa2 * Pa1.inverse() is not identity.
	 *
	 * If the above conditions are not fulfilled or only marginally so, either the scale numerator
	 * or denominator will be exactly or close to zero. In either case, the scale estimate is likely
	 * to be invalid or poor.
	 */
    bool estimateMinimal(const Vector2d& yb1, const Vector2d& yb2,
		double& scale_numerator, double& scale_denominator) const;

	/* Estimate the scale of the essential matrix translation component.
	*
	* @param yb1n Vector of K-normalized 2D points observed in Pb1
	* @param yb2n Vector of K-normalized 2D points observed at time 2, corresponding to those seen at time 1.
	*
	* @param scale	      [Output] The estimated scale.
	* @param num_inliers  [Output] Number inliers with the returned scale.
	* @param num_iters    [Output] Number of iterations in the RANSAC loop
	*
	* @returns true on success, false if no good estimate could be found.
	*/
    bool estimate(const std::vector<Vector2d>& yb1s, const std::vector<Vector2d>& yb2s, double& scale,
		uint& num_inliers, uint& num_iters);
};

struct ESPoseEstimator : public EssentialMatrixSolverBase
{
	EssentialMatrixSolver::Parameters es_params;
	EssentialMatrixScaleSolver::Parameters ss_params;
	float flen_c;

	ESPoseEstimator(const float flen_a, const float flen_b, const float flen_c);

	static EssentialMatrixSolver::Parameters getDefaultParamsE(float focal_length);
	static EssentialMatrixScaleSolver::Parameters getDefaultParamsScale(float focal_length);

    PoseD estimatePose(const PoseD& Tba, const PoseD& Tca,
        const std::vector<Vector2d>& ya1s, const std::vector<Vector2d>& ya2s,
        const std::vector<Vector2d>& yb1s, const std::vector<Vector2d>& yb2s,
        const std::vector<Vector2d>& yc1s, const std::vector<Vector2d>& yc2s,
		std::vector<std::vector<bool>>& inliers, uint& num_inliers, uint& num_iters);
};
}
