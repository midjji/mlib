#pragma once
#include <vector>
#include <mlib/utils/cvl/pose.h>






struct MotionTester
{
	struct Parameters
	{
		float focal_length;				//!< Camera focal length
		uint min_num_points;			//!< Minimum number of points/correspondences required 

		float corner_noise_stddev;		//!< Expected corner location noise standard deviation [pixels]
		float min_disparity_sigma;		//!< Minimum disparity threshold, relative to the corner noise standard deviation [pixels]
		float outlier_sigma;			//!< Outlier-rejection threshold, relative to the standard deviation of all disparities.
		float test1_survivor_ratio;		//!< Minimal ratio of correspondences surviving test 1.

		static Parameters getDefault(float focal_length)
		{
			Parameters p;

			p.focal_length = focal_length;
			p.min_num_points = 50;

			p.corner_noise_stddev = 1.6;

            p.min_disparity_sigma = 2.0f;// 0.95
            p.outlier_sigma = 3.0f; // 0.998
			p.test1_survivor_ratio = 0.25f;

			return p;
		}
	};

	Parameters parameters;

	const std::vector<cvl::Vector2d>& y1s; //!< K-normalized 2D observations at t = 1
	const std::vector<cvl::Vector2d>& y2s; //!< K-normalized 2D observations at t = 2, corresponding to those at t = 1

	const uint n;	//!< Number of correspondences

	std::vector<cvl::Vector2d> mvectors; //!< Motion vectors from last computeDisparities()
	std::vector<float> disparities; //!< Motion disparities (in pixels) from last computeDisparities()

	MotionTester(const std::vector<cvl::Vector2d>& y1s, const std::vector<cvl::Vector2d>& y2s);

	/** Compute disparities, optionally compensating for camera rotation
	* @param P21  Optional: Relative camera transform. If nullptr, zero rotation is assumed
	*/
    void computeDisparities(const cvl::PoseD* P21 = nullptr);

	/// Check if the motion is good enough for e.g E-matrix estimation
	bool testDisparities();

	/// Check for any motion. Combines computeDisparities() and testDisparities()
	bool haveMotion();

	/// Check for translational motion, compensating for camera rotation.  Combines computeDisparities() and testDisparities()
    bool haveTranslation(const cvl::PoseD& P21);


};
