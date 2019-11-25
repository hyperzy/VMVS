//
// Created by himalaya on 11/22/19.
//

#include "velocity_calculator.h"
#include <iostream>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;


PhiCalculator::PhiCalculator(const std::vector<Camera> &all_cams, const std::vector<Vec3> &coord,
        DimUnit height, DimUnit width, DimUnit depth, dtype resolution, unsigned short half_window_size):
                                                        all_cams(all_cams), coord(coord.data()), height(height),
                                                        width(width), depth(depth), phi(nullptr), psi_arr(nullptr),
                                                        xy_normal(Vec3(0, 0, 1)), resolution(resolution),
                                                        half_window_size(half_window_size)

{
    dtype step_size = resolution / 4;
    for (short u = -half_window_size; u <= half_window_size; u++) {
        for (short v = -half_window_size; v <= half_window_size; v++) {
            tangent_plane_points_uv.emplace_back(Vec3(u * step_size, v * step_size, 0));
        }
    }
}

void PhiCalculator::Set_phi(const std::vector<dtype> &phi)
{
    this->phi =  phi.data();
}

void PhiCalculator::Set_psi(const std::vector<Visibility> &vis_arr)
{
    this->psi_arr = vis_arr.data();
}

dtype PhiCalculator::Compute_discrepancy(IdxType i, IdxType j, IdxType k, bool inside)
{
    auto idx_ijk = Index(i, j, k);
    Vec3 given_point = coord[idx_ijk];
    // normal vector of tangent plane at given point
    Vec3 point_normal;
    Compute_normal(point_normal, i, j, k);
    // rotation matrix from (0, 0, 1) to normal_point to find coordinates of the points on the tangent plane
    Mat rot_mat;
    Compute_rotation(rot_mat, xy_normal, point_normal);
    vector<Vec<dtype, 4>> tangent_plane_points;
    tangent_plane_points.reserve(pow(half_window_size * 2 + 1, 2));
    Compute_tangent_plane_points(tangent_plane_points, i, j, k, rot_mat);
//    test(tangent_plane_poitns, point_normal);
    // computing
    vector<unsigned short> visible_cam_idx;
    dtype sum_Phi = 0;
    for (unsigned short cam_idx = 0; cam_idx < all_cams.size(); cam_idx++) {
        if (psi_arr[cam_idx].psi[idx_ijk] > 0) {
            for (auto & iter : visible_cam_idx) {
                sum_Phi += Compute_discrepancy_ij(cam_idx, iter, tangent_plane_points);
            }
            visible_cam_idx.emplace_back(cam_idx);
        }
    }
    if (visible_cam_idx.size() < 2) return MAX_NCC;
    else
        // we use '-1' instead of '+1' because we count an term for a pair of visible cameras
        return 2 * sum_Phi / (visible_cam_idx.size() * (visible_cam_idx.size() - 1.));
}

//
Mat Skew_symmetric(const Vec3 &input_vec)
{
    Mat ssc = Mat::zeros(3, 3, DTYPE);
    ssc.at<dtype>(0, 1) = -input_vec[2];
    ssc.at<dtype>(0, 2) = input_vec[1];
    ssc.at<dtype>(1, 0) = input_vec[2];
    ssc.at<dtype>(1, 2) = -input_vec[0];
    ssc.at<dtype>(2, 0) = -input_vec[1];
    ssc.at<dtype>(2, 1) = input_vec[0];
//    cout << ssc << endl;
    return ssc;
}
void PhiCalculator::Compute_rotation(cv::Mat &rot_mat, const Vec3 &src_vec, const Vec3 &tar_vec)
{
    dtype float_err = 1e-7;
    Vec3 v = src_vec.cross(tar_vec);
    // skew-symmetric cross product matrix of v
    Mat ssc = Skew_symmetric(v);
//    cout << ssc * ssc << endl;
    // sinusoid of the angle from src_vec to tar_vec
    dtype sine_angle = norm(v, NORM_L2);
    dtype cos_angle = src_vec.dot(tar_vec);
    if (abs(sine_angle) > float_err)
        rot_mat = Mat::eye(3, 3, DTYPE) + ssc + ssc * ssc * (1 - cos_angle) / sine_angle / sine_angle;
    else
        rot_mat = sine_angle >= 0 ? Mat::eye(3, 3, DTYPE) : -Mat::eye(3, 3, DTYPE);
//    cout << rot_mat << endl;
}

void PhiCalculator::Compute_normal(Vec3 &normal_vec, IdxType i, IdxType j, IdxType k)
{
    normal_vec[0] = (phi[Index(i + 1, j, k)] - phi[Index(i - 1, j, k)]) / 2;
    normal_vec[1] = (phi[Index(i, j + 1, k)] - phi[Index(i, j - 1, k)]) / 2;
    normal_vec[2] = (phi[Index(i, j, k + 1)] - phi[Index(i, j, k - 1)]) / 2;
    normal_vec /= norm(normal_vec, NORM_L2);
}

dtype PhiCalculator::Compute_discrepancy_ij(IdxType cam_1_idx, IdxType cam_2_idx, const std::vector<cv::Vec<dtype, 4>> &points)
{
    int size = (half_window_size + half_window_size + 1) * (half_window_size + half_window_size + 1);
    vector<dtype> v1, v2;
    v1.reserve(size);
    v2.reserve(size);
    for (int i = 0; i < points.size(); i++) {
        // projection point on camera1's plane
        Mat res1 = all_cams[cam_1_idx].P * points[i];
        Mat res2 = all_cams[cam_2_idx].P * points[i];
        int x1 = cvRound(res1.at<dtype>(0, 0) / res1.at<dtype>(0, 2));
        int y1 = cvRound(res1.at<dtype>(0, 1) / res1.at<dtype>(0, 2));
        int x2 = cvRound(res2.at<dtype>(0, 0) / res2.at<dtype>(0, 2));
        int y2 = cvRound(res2.at<dtype>(0, 1) / res2.at<dtype>(0, 2));
        // if the projection is outside the image range, than set Phi_ij as 2, which is uncorrelated
        if (x1 < 0 || x1 >= all_cams[0].gray_img.cols || y1 < 0 || y1 >= all_cams[0].gray_img.rows
            || x2 < 0 || x2 >= all_cams[0].gray_img.cols || y2 < 0 || y2 >= all_cams[0].gray_img.rows) {
            return MAX_NCC;
        }
//        cout << x1 << ", " << y1 << endl;

        v1.emplace_back(all_cams[cam_1_idx].gray_img.at<uchar>(y1, x1));
        v2.emplace_back(all_cams[cam_2_idx].gray_img.at<uchar>(y2, x2));
    }
//    auto t_mean1 = cv::mean(m_v1);
//    auto t_mean2 = cv::mean(m_v2);
    dtype mean1 = accumulate(v1.begin(), v1.end(), 0.) / size;
    dtype mean2 = accumulate(v2.begin(), v2.end(), 0.) / size;
    // standardize the data
    for (auto &iter : v1) {
        iter -= mean1;
    }
    for (auto &iter : v2) {
        iter -= mean2;
    }
    // compute
    dtype self_corr1 = 0, self_corr2 = 0, cross_term = 0;
    for (auto &iter : v1) {
        self_corr1 += iter * iter;
    }
    for (auto &iter : v2) {
        self_corr2 += iter * iter;
    }
    for (int i = 0; i < v1.size(); i++) {
        cross_term += v1[i] * v2[i];
    }
    // if self correlation item is 0, then correlation coefficient is 0
    if (self_corr1 == 0 || self_corr2 == 0) {
        return 1.;
    }
    else
        return 1. - cross_term / sqrt(self_corr1 * self_corr2);
}

void PhiCalculator::Compute_tangent_plane_points(std::vector<cv::Vec<dtype, 4>> &tangent_plane_points, IdxType i, IdxType j, IdxType k, const cv::Mat &rot_mat)
{
    // step size for small grid in tangent patch
    unsigned long idx_ijk = Index(i, j, k);
//    dtype point_x = coord[idx_ijk][0];
//    dtype point_y = coord[idx_ijk][1];
//    dtype point_z = coord[idx_ijk][2];

    for (const auto &iter : tangent_plane_points_uv) {
        Mat res = rot_mat * iter;
//        cout << res << endl;
//        cout << Vec3(res) << endl;
        Vec<dtype, 4> hom_res(coord[idx_ijk][0] + res.at<dtype>(0, 0),
                              coord[idx_ijk][1] + res.at<dtype>(0, 1),
                              coord[idx_ijk][2] + res.at<dtype>(0, 2),
                              1);
//        cout << hom_res << endl;
        tangent_plane_points.emplace_back(hom_res);
    }

}

// test if the tangent plane points are computed correctly
void test(const std::vector<Vec3> &tangent_plane_points, const Vec3 &point_normal)
{
    for (int i = 1; i < tangent_plane_points.size(); i++) {
        cout << (tangent_plane_points[i] - tangent_plane_points[0]).dot(point_normal) << endl;
    }
}



