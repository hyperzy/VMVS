//
// Created by himalaya on 11/22/19.
//

#ifndef VMVS_VELOCITY_CALCULATOR_H
#define VMVS_VELOCITY_CALCULATOR_H

#include "base.h"
#include "visibility.h"

#define USE_SIL 1

class PhiCalculator {
public:
    PhiCalculator(const std::vector<Camera> &all_cams, const std::vector<Vec3> &coord,
                    DimUnit height, DimUnit width, DimUnit depth,
                    dtype resolution, unsigned short half_window_size = 2);
    /**
     * @brief Set the phi pointer
     * @param phi STL vector storing phi val
     */
    void Set_phi(const std::vector<dtype> &phi);

    /**
     * @brief Set the visibility array pointer
     * @param vis_arr STL vector storing visibility for each camera
     */
    void Set_psi(const std::vector<Visibility> &vis_arr);

    dtype Compute_discrepancy(IdxType i, IdxType j, IdxType k, bool inside);

private:
    bool isValidRange(IdxType i, IdxType j, IdxType k) const;

    static void Compute_rotation(cv::Mat &rot_mat, const Vec3 &src_vec, const Vec3 &tar_vec);

    /**
     * @brief Use central difference to approximate the unit normal
     * @param normal_vec
     * @param idx_ijk
     */
    void Compute_normal(Vec3 &normal_vec, IdxType i, IdxType j, IdxType k);

    /**
     * @brief Calculating the discrepancy between camera i and camera j
     * @param cam_1_idx
     * @param cam_2_idx
     * @return
     */
    dtype Compute_discrepancy_ij(IdxType cam_1_idx, IdxType cam_2_idx, const std::vector<cv::Vec<dtype, 4>> &points);

    /**
     * @breif Compute all the points needed on the tangent plane
     * @param i
     * @param j
     * @param k
     * @param rot_mat
     */
    void Compute_tangent_plane_points(std::vector<cv::Vec<dtype, 4>> &tangent_plane_points, IdxType i, IdxType j, IdxType k, const cv::Mat &rot_mat);


    const dtype resolution;
    const unsigned short half_window_size;
    unsigned long Index(IdxType i, IdxType j, IdxType k) const;
    DimUnit height, width, depth;
    const std::vector<Camera> &all_cams;
    // here we used const pointer instead onf constant reference since there is a swap clause in FMM3d
    Vec3 const *coord;
    const dtype *phi;
    const Visibility *psi_arr;
    // normal vector of x-y plane
    const Vec3 xy_normal;
    // tangent plane points in uv coordinates, whose z entry are 0
    std::vector<Vec3> tangent_plane_points_uv;
    // largest value of normalized cross-correlation item
    const dtype MAX_NCC = 2;
    const dtype MIN_NCC = 0;
};

inline unsigned long PhiCalculator::Index(IdxType i, IdxType j, IdxType k) const
{
    return i * this->width * this->depth + j * this->depth + k;
}

inline bool PhiCalculator::isValidRange(IdxType i, IdxType j, IdxType k) const
{
    return (i >= 0 && i < this->height && j >= 0 && j < this->width
            && k >= 0 && k < this->depth);
}

void test(const std::vector<Vec3> &tangent_plane_points, const Vec3 &point_normal);
#endif //VMVS_VELOCITY_CALCULATOR_H
