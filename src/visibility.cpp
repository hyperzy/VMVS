//
// Created by himalaya on 10/31/19.
//

#include "visibility.h"
#include <iostream>
#include <cmath>
#include <omp.h>
//#define NDEBUG
#include <cassert>

using namespace std;
using namespace cv;


Visibility::Visibility(const std::vector<dtype> &extent, const Camera &camera, const std::vector<Vec3> &coord,
                            dtype resolution, dtype height, dtype width, dtype depth):
                        extent(extent), camera(camera), resolution(resolution), coord(coord),
                        height(height), width(width), depth(depth), cam_coord(camera.t), phi(nullptr)
{
    // 1% error tolerant for bilinear interpolation
    this->err = 0.01;
    this->tolerance = err * resolution;
    this->psi.resize(height * width * depth);
    this->vertices.reserve(8);
    // (xmin, ymin, zmin)
    this->vertices.emplace_back(Vec3(extent[0], extent[2], extent[4]));
    // (xmax, ymin, zmin)
    this->vertices.emplace_back(Vec3(extent[1], extent[2], extent[4]));
    this->vertices.emplace_back(Vec3(extent[1], extent[3], extent[4]));
    this->vertices.emplace_back(Vec3(extent[0], extent[3], extent[4]));
    this->vertices.emplace_back(Vec3(extent[0], extent[2], extent[5]));
    this->vertices.emplace_back(Vec3(extent[1], extent[2], extent[5]));
    this->vertices.emplace_back(Vec3(extent[1], extent[3], extent[5]));
    this->vertices.emplace_back(Vec3(extent[0], extent[3], extent[5]));
    this->dir = this->Determine_direction();
    this->float_err = 1e-6;
}

void Visibility::Set_phi(const std::vector<dtype> &phi_data)
{
    if (this->phi == nullptr)
        this->phi = phi_data.data();
}

bool Visibility::isPositionHere(const Vec3 &p1, const Vec3 &plane_point, const Vec3& plane_normal)
{
    return Vec3(p1 - plane_point).dot(plane_normal) >= 0;
}

int Visibility::Determine_direction() const
{
    Vec3 normalA(1, -1, 0);
    Vec3 normalB(0, -1, 0);
    Vec3 normalC(-1, -1, 0);
    Vec3 normalD(1, 0, 0);

//    cout << cam_coord << endl;
//    cout << camera.t << endl;
    /*
     *     \          /
     *      \   3    /
     *       --------
     *   3   |      |  2
     *       |      |
     *       --------
     *      /        \
     *     /    1     \
     *
     *   there are four regions
     */
    // todo: for now, since we know there is no camera on the top or bottom of the object,
    //  we only consider four cases. Another two cases remain to be done
    if (isPositionHere(cam_coord, this->vertices[0], normalA) && isPositionHere(cam_coord, this->vertices[0], normalB)
        && isPositionHere(cam_coord, this->vertices[1], normalC))
        return PlaneDirection::XZ_POS;
    else if (isPositionHere(cam_coord, this->vertices[1], -normalC) && isPositionHere(cam_coord, this->vertices[1], normalD)
                && isPositionHere(cam_coord, this->vertices[2], normalA))
        return PlaneDirection::YZ_NEG;
    else if (isPositionHere(cam_coord, this->vertices[2], -normalA) && isPositionHere(cam_coord, this->vertices[2], -normalB)
                && isPositionHere(cam_coord, this->vertices[3], -normalC))
        return PlaneDirection::XZ_NEG;
    else if (isPositionHere(cam_coord, this->vertices[3], normalC) && isPositionHere(cam_coord, this->vertices[3], -normalD)
                && isPositionHere(cam_coord, this->vertices[0], -normalA))
        return PlaneDirection::YZ_POS;
    else {
        cerr << "Error Camera Position. Try to do coordinate system transformation first" << endl;
        exit(1);
    }
}

bool Visibility::isOnGridPoint() const
{
    // todo: only two cases are considered for now
    if (this->dir == PlaneDirection::XZ_POS || this->dir == PlaneDirection::XZ_NEG) {
        dtype f_index_x = (cam_coord[0] - extent[0]) / resolution;
        dtype f_index_z = (cam_coord[2] - extent[4]) / resolution;
        return f_index_x == floor(f_index_x) && f_index_z == floor(f_index_z);
    }
    else if (this->dir == PlaneDirection::YZ_POS || this->dir == PlaneDirection::YZ_NEG) {
        dtype f_index_y = (cam_coord[1] - extent[2]) / resolution;
        dtype f_index_z = (cam_coord[2] - extent[4]) / resolution;
        return f_index_y == floor(f_index_y) && f_index_z == floor(f_index_z);
    }
}



void Visibility::Calculate_all()
{
//    std::fill(psi.begin(), psi.end(), 0);
    // first determine the camera position w.r.t grid index
    dtype f_index_x = (cam_coord[0] - extent[0]) / resolution;
    dtype f_index_y = (cam_coord[1] - extent[2]) / resolution;
    dtype f_index_z = (cam_coord[2] - extent[4]) / resolution;

    // this step is used in case float accuracy issue happens
    // e.g. 3.9999999999 should be just 4. Otherwise, after floor function, we will get 3
    if (fabs(f_index_x - round(f_index_x)) < float_err)
        f_index_x = dtype(round(f_index_x));
    if (fabs(f_index_y - round(f_index_y)) < float_err)
        f_index_y = dtype(round(f_index_y));
    if (fabs(f_index_z - round(f_index_z)) < float_err)
        f_index_z = dtype(round(f_index_z));

    //// For this case, we will sweep X-Z plane
    if (this->dir == PlaneDirection::XZ_POS || this->dir == PlaneDirection::XZ_NEG) {
        //// 1. initialize the visibility value of four closest surface arount the camera's
        //// projection on the extent surface
        bool direction_flag = this->dir == PlaneDirection::XZ_POS;
        auto smaller_idx_x = (IdxType)floor(f_index_x);
        // if the projection is outside the bounding box
        smaller_idx_x = smaller_idx_x >= height ? height - 2 : (smaller_idx_x < 0 ? 0 : smaller_idx_x);
        IdxType  larger_idx_x = smaller_idx_x + 1;
        auto smaller_idx_z = (IdxType)floor(f_index_z);
        smaller_idx_z = smaller_idx_z >= depth ? depth - 2 : (smaller_idx_z < 0 ? 0 : smaller_idx_z);
        IdxType  larger_idx_z = smaller_idx_z + 1;
        // assign initial value to the first sweeping plane(since all the points on it are visible)
        IdxType start_j = (direction_flag ? 0 : width - 1);
        for (IdxType i = 0; i < height; i++) {
            for (IdxType k = 0; k < depth; k++) {
                assert(isValidRange(i, start_j, k));
                psi[Index(i, start_j, k)] = phi[Index(i, start_j, k)];
            }
        }
        vector<IdxType> x_iter_arr{smaller_idx_x, larger_idx_x};
        vector<IdxType> z_iter_arr{smaller_idx_z, larger_idx_z};

        // Besides, if the camera's projection exceeds the box boundary, we need to assign value to other box
        // surfaces since they are visible
        bool x_upper_overflow = false, x_lower_overflow = false;
        bool z_upper_overflow = false, z_lower_overflow = false;
        if (f_index_x > height - 1) {
            x_upper_overflow = true;
            IdxType i = height - 1;
            for (IdxType j = 0; j < width; j++) {
                for (IdxType k = 0; k < depth; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            x_iter_arr.resize(1);
            x_iter_arr[0] = smaller_idx_x;
        }
        else if (f_index_x < 0) {
            x_lower_overflow = true;
            IdxType i = 0;
            for (IdxType j = 0; j < width; j++) {
                for (IdxType k = 0; k < depth; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            x_iter_arr.resize(1);
            x_iter_arr[0] = larger_idx_x;
        }

        if (f_index_z > depth - 1) {
            z_upper_overflow = true;
            IdxType k = depth - 1;
            for (IdxType i = 0; i < height; i++) {
                for (IdxType j = 0; j < width; j++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            z_iter_arr.resize(1);
            z_iter_arr[0] = smaller_idx_z;
        }
        else if (f_index_z < 0) {
            z_lower_overflow = true;
            IdxType k = 0;
            for (IdxType i = 0; i < height; i++) {
                for (IdxType j = 0; j < width; j++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            z_iter_arr.resize(1);
            z_iter_arr[0] = larger_idx_z;
        }

        // Since we used floor() + 1 instead of ceil(), camera's projection on grid case is
        // generalized as not_on_grid() case
        int increment = direction_flag ? 1 : -1;
        IdxType y_bound = direction_flag ? width : 0;
        for (int j = (direction_flag ? 1 : width - 2); (direction_flag ? j < y_bound : j >= y_bound); j += increment) {
            IdxType x_upper_bound = x_upper_overflow ? height - 1 : height;
            for (const auto &k : z_iter_arr) {
                for (IdxType i = (x_lower_overflow ? 1 : 0); i < x_upper_bound; i++) {
                    psi[Index(i, j, k)] = Calculate_point(i, j, k);
                }
            }
//            IdxType z_upper_bound = z_upper_overflow ? depth - 1 : depth;
            // todo: since we know there are no camera above or below the object, for now I do not
            //  consider the complex case which is the upper or lower surface is initialized
            for (const auto &i : x_iter_arr) {
                for (IdxType k = 0; k < smaller_idx_z; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = Calculate_point(i, j, k);
                }
                for (IdxType k = larger_idx_z + 1; k < depth; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = Calculate_point(i, j, k);
                }
            }
        }
        cout << "finished initial surface" << endl;

        //// 2. compute visibility of four quadrants
        for (int j = (direction_flag ? 1 : width - 2); (direction_flag ? j < y_bound : j >= y_bound); j += increment) {
            for (int s1 = -1; s1 <= 1; s1 += 2) {
                for (int s2 = -1; s2 <= 1; s2 += 2) {
                    IdxType x_bound = s1 < 0 ? (x_lower_overflow ? 1 : 0) : (x_upper_overflow ? height - 1 : height);
                    IdxType z_bound = s2 < 0 ? (z_lower_overflow ? 1 : 0) : (z_upper_overflow ? depth - 1 : depth);
                    for (int i = (s1 < 0 ? smaller_idx_x - 1 : larger_idx_x + 1); (s1 < 0 ? i >= x_bound : i < x_bound); i += s1) {
                        for (int k = (s2 < 0 ? smaller_idx_z - 1 : larger_idx_z + 1); (s2 < 0 ? k >= z_bound : k < z_bound); k += s2) {
                            assert(isValidRange(i, j, k));
                            psi[Index(i, j, k)] = Calculate_point(i, j, k);
                        }
                    }
                }
            }
        }
    }
    else if (this->dir == PlaneDirection::YZ_POS || this->dir == PlaneDirection::YZ_NEG) {
        bool direction_flag = this->dir == YZ_POS;
        //// 1. initialize the visibility value of four closest surface arount the camera's
        //// projection on the extent surface
        auto smaller_idx_y = (IdxType)floor(f_index_y);
        // if the projection is outside the bounding box
        smaller_idx_y = smaller_idx_y >= width ? width - 2 : (smaller_idx_y < 0 ? 0 : smaller_idx_y);
        IdxType larger_idx_y = smaller_idx_y + 1;
        auto smaller_idx_z = (IdxType)floor(f_index_z);
        smaller_idx_z = smaller_idx_z >= depth ? depth - 2 : (smaller_idx_z < 0 ? 0 : smaller_idx_z);
        IdxType  larger_idx_z = smaller_idx_z + 1;
        // assign initial value to the first sweeping plane(since all the points on it are visible)
        IdxType start_i = (direction_flag ? 0 : height - 1);
        for (IdxType j = 0; j < width; j++) {
            for (IdxType k = 0; k < depth; k++) {
                assert(isValidRange(start_i, j, k));
                psi[Index(start_i, j, k)] = phi[Index(start_i, j, k)];
            }
        }
        vector<IdxType> y_iter_arr{smaller_idx_y, larger_idx_y};
        vector<IdxType> z_iter_arr{smaller_idx_z, larger_idx_z};

        // Besides, if the camera's projection exceeds the box boundary, we need to assign value to other box
        // surfaces since they are visible
        bool y_upper_overflow = false, y_lower_overflow = false;
        bool z_upper_overflow = false, z_lower_overflow = false;
        if (f_index_y > width - 1) {
            y_upper_overflow = true;
            IdxType j = width - 1;
            for (IdxType i = 0; i < height; i++) {
                for (IdxType k = 0; k < depth; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            y_iter_arr.resize(1);
            y_iter_arr[0] = smaller_idx_y;
        }
        else if (f_index_y < 0) {
            y_lower_overflow = true;
            IdxType j = 0;
            for (IdxType i = 0; i < height; i++) {
                for (IdxType k = 0; k < depth; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            y_iter_arr.resize(1);
            y_iter_arr[0] = larger_idx_y;
        }

        if (f_index_z > depth - 1) {
            z_upper_overflow = true;
            IdxType k = depth - 1;
            for (IdxType i = 0; i < height; i++) {
                for (IdxType j = 0; j < width; j++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            z_iter_arr.resize(1);
            z_iter_arr[0] = smaller_idx_z;
        }
        else if (f_index_z < 0) {
            z_lower_overflow = true;
            IdxType k = 0;
            for (IdxType i = 0; i < height; i++) {
                for (IdxType j = 0; j < width; j++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = phi[Index(i, j, k)];
                }
            }
            z_iter_arr.resize(1);
            z_iter_arr[0] = larger_idx_z;
        }

        // Since we used floor() + 1 instead of ceil(), camera's projection on grid case is
        // generalized as not_on_grid() case
        int increment = direction_flag ? 1 : -1;
        IdxType x_bound = direction_flag ? height : 0;
        for (int i = (direction_flag ? 1 : height - 2); (direction_flag ? i < x_bound : i >= x_bound); i += increment) {
            IdxType y_upper_bound = y_upper_overflow ? width - 1 : width;
            for (const auto &k : z_iter_arr) {
                for (IdxType j = (y_lower_overflow ? 1 : 0); j < y_upper_bound; j++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = Calculate_point(i, j, k);
                }
            }
//            IdxType z_upper_bound = z_upper_overflow ? depth - 1 : depth;
            // todo: since we know there are no camera above or below the object, for now I do not
            //  consider the complex case which is the upper or lower surface is initialized
            for (const auto &j : y_iter_arr) {
                for (IdxType k = 0; k < smaller_idx_z; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = Calculate_point(i, j, k);
                }
                for (IdxType k = larger_idx_z + 1; k < depth; k++) {
                    assert(isValidRange(i, j, k));
                    psi[Index(i, j, k)] = Calculate_point(i, j, k);
                }
            }
        }
        cout << "finished initial surface" << endl;

        //// 2. compute visibility of four quadrants
        for (int i = (direction_flag ? 1 : height - 2); (direction_flag ? i < x_bound : i >= x_bound); i += increment) {
            for (int s1 = -1; s1 <= 1; s1 += 2) {
                for (int s2 = -1; s2 <= 1; s2 += 2) {
                    IdxType y_bound = s1 < 0 ? (y_lower_overflow ? 1 : 0) : (y_upper_overflow ? width - 1 : width);
                    IdxType z_bound = s2 < 0 ? (z_lower_overflow ? 1 : 0) : (z_upper_overflow ? depth - 1 : depth);
                    for (int j = (s1 < 0 ? smaller_idx_y - 1 : larger_idx_y + 1); (s1 < 0 ? j >= y_bound : j < y_bound); j += s1) {
                        for (int k = (s2 < 0 ? smaller_idx_z - 1 : larger_idx_z + 1); (s2 < 0 ? k >= z_bound : k < z_bound); k += s2) {
                            assert(isValidRange(i, j, k));
                            psi[Index(i, j, k)] = Calculate_point(i, j, k);
                        }
                    }
                }
            }
        }
    }
    else {
        cerr << "wrong direction for now" << endl;
        exit(1);
    }
}

dtype Visibility::Calculate_point(IdxType i, IdxType j, IdxType k)
{
    dtype minval = phi[Index(i, j, k)];
    dtype temp_val = 0;
    Vec3 vec = cam_coord - coord[Index(i, j, k)];
    IdxType argmax_axis = 0;
    dtype max = -1;
    for (int i = 0; i < 3; i++) {
        if (fabs(vec[i]) < float_err) vec[i] = 0;
        if (fabs(vec[i]) > max) {
            argmax_axis = i;
            max = fabs(vec[i]);
        }
    }

    int x_dir_increment = vec[0] > 0 ? 1 : (vec[0] < 0 ? -1 : 0);
    int y_dir_increment = vec[1] > 0 ? 1 : (vec[1] < 0 ? -1 : 0);
    int z_dir_increment = vec[2] > 0 ? 1 : (vec[2] < 0 ? -1 : 0);

    IdxType cur_x_index, cur_y_index, cur_z_index;
    auto cur_coord = coord[Index(i, j, k)];

    if (argmax_axis == 0) {
//            dtype cur_x_coord = cur_coord[0];
//            dtype cur_y_coord = cur_coord[1];
//            dtype cur_z_coord = cur_coord[2];
        assert(x_dir_increment != 0);
        assert(isValidRange(i + x_dir_increment, j, k));
        auto next_coord = Intersect(vec, cur_coord, coord[Index(i + x_dir_increment, j, k)], PlaneDirection::YZ_POS);

        IdxType y_changed = j + y_dir_increment;
        IdxType z_changed = k + z_dir_increment;
        // then determine if the ray cross other planes instead of X := x+x_dir*resolution
        assert(isValidRange(i, y_changed, k));
        Vec3 determiner_coord_for_y = coord[Index(i, y_changed, k)];
        assert(isValidRange(i, j, z_changed));
        Vec3 determiner_coord_for_z = coord[Index(i, j, z_changed)];
        // then determine if the ray firstly intersect the Y = a surface or Z = b surface before intersecting with X = c. a, b, c
        // are some constants
        Vec3 next_coord_cand1(next_coord);
        Vec3 next_coord_cand2(next_coord);
        bool flag_cand1 = false, flag_cand2 = false;
        if ((y_dir_increment > 0 && next_coord[1] > determiner_coord_for_y[1])
            || (y_dir_increment < 0 && next_coord[1] < determiner_coord_for_y[1])) {
            next_coord_cand1 = Intersect(vec, cur_coord, determiner_coord_for_y, PlaneDirection::XZ_POS);
            flag_cand1 = true;
        }
        if ((z_dir_increment > 0 && next_coord[2] > determiner_coord_for_z[2])
            || (z_dir_increment < 0 && next_coord[2] < determiner_coord_for_z[2])) {
            next_coord_cand2 = Intersect(vec, cur_coord, determiner_coord_for_z, PlaneDirection::XY_POS);
            flag_cand2 = true;
        }
        // determine the increment of the index of four surfaces points
        int surface_dir = PlaneDirection::YZ_POS;
        if ((x_dir_increment > 0 ? next_coord_cand1[0] < next_coord_cand2[0] : next_coord_cand1[0] > next_coord_cand2[0])
            && flag_cand1) {
            next_coord = next_coord_cand1;
            surface_dir = XZ_POS;
        }
        else if ((x_dir_increment > 0 ? next_coord_cand2[0] < next_coord_cand1[0] : next_coord_cand2[0] > next_coord_cand1[0])
                 && flag_cand2) {
            next_coord = next_coord_cand2;
            surface_dir = XY_POS;
        }
        switch (surface_dir) {
            case YZ_POS:
                assert(isValidRange(i + x_dir_increment, j, k) && isValidRange(i + x_dir_increment, j + y_dir_increment, k)
                        && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                        && isValidRange(i + x_dir_increment, j, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i + x_dir_increment, j, k),
                                                     Index(i + x_dir_increment, j + y_dir_increment, k),
                                                     Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                                     Index(i + x_dir_increment, j, k + z_dir_increment), surface_dir);
                break;
            case XZ_POS:
                assert(isValidRange(i, j + y_dir_increment, k) && isValidRange(i + x_dir_increment, j + y_dir_increment, k)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i, j + y_dir_increment, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i, j + y_dir_increment, k),
                                                     Index(i + x_dir_increment, j + y_dir_increment, k),
                                                     Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                                     Index(i, j + y_dir_increment, k + z_dir_increment), surface_dir);
                break;
            case XY_POS:
                assert(isValidRange(i, j, k + z_dir_increment) && isValidRange(i, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i, j, k + z_dir_increment),
                                                     Index(i, j + y_dir_increment, k + z_dir_increment),
                                                     Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                                     Index(i + x_dir_increment, j, k + z_dir_increment), surface_dir);
                break;
            default:
                cerr << "wrong interpolation direction" << endl;
                exit(1);
        }

    }
    else if (argmax_axis == 1) {
        assert(y_dir_increment != 0);
        assert(isValidRange(i, j + y_dir_increment, k));
        auto next_coord = Intersect(vec, cur_coord, coord[Index(i, j + y_dir_increment, k)], PlaneDirection::XZ_POS);

        IdxType x_changed = i + x_dir_increment;
        IdxType z_changed = k + z_dir_increment;
        // then determine if the ray cross other planes instead of X := x+x_dir*resolution
        assert(isValidRange(x_changed, j, k));
        Vec3 determiner_coord_for_x = coord[Index(x_changed, j, k)];
        assert(isValidRange(i, j, z_changed));
        Vec3 determiner_coord_for_z = coord[Index(i, j, z_changed)];
        // then determine if the ray firstly intersect the Y = a surface or Z = b surface before intersecting with X = c. a, b, c
        // are some constants
        Vec3 next_coord_cand1(next_coord);
        Vec3 next_coord_cand2(next_coord);
        bool flag_cand1 = false, flag_cand2 = false;
        if ((x_dir_increment > 0 && next_coord[0] > determiner_coord_for_x[0])
            || (x_dir_increment < 0 && next_coord[0] < determiner_coord_for_x[0])) {
            next_coord_cand1 = Intersect(vec, cur_coord, determiner_coord_for_x, PlaneDirection::YZ_POS);
            flag_cand1 = true;
        }
        if ((z_dir_increment > 0 && next_coord[2] > determiner_coord_for_z[2])
            || (z_dir_increment < 0 && next_coord[2] < determiner_coord_for_z[2])) {
            next_coord_cand2 = Intersect(vec, cur_coord, determiner_coord_for_z, PlaneDirection::XY_POS);
            flag_cand2 = true;
        }
        // determine the increment of the index of four surfaces points
        int surface_dir = PlaneDirection::XZ_POS;
        if ((y_dir_increment > 0 ? next_coord_cand1[1] < next_coord_cand2[1] : next_coord_cand1[1] > next_coord_cand2[1])
            && flag_cand1) {
            next_coord = next_coord_cand1;
            surface_dir = YZ_POS;
        }
        else if ((y_dir_increment > 0 ? next_coord_cand2[1] < next_coord_cand1[1] : next_coord_cand2[1] > next_coord_cand1[1])
                 && flag_cand2) {
            next_coord = next_coord_cand2;
            surface_dir = XY_POS;
        }
        switch (surface_dir) {
            case YZ_POS:
                assert(isValidRange(i + x_dir_increment, j, k) && isValidRange(i + x_dir_increment, j + y_dir_increment, k)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i + x_dir_increment, j, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                         Index(i + x_dir_increment, j, k + z_dir_increment), surface_dir);
                break;
            case XZ_POS:
                assert(isValidRange(i, j + y_dir_increment, k) && isValidRange(i + x_dir_increment, j + y_dir_increment, k)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i, j + y_dir_increment, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i, j + y_dir_increment, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                         Index(i, j + y_dir_increment, k + z_dir_increment), surface_dir);
                break;
            case XY_POS:
                assert(isValidRange(i, j, k + z_dir_increment) && isValidRange(i, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i, j, k + z_dir_increment),
                                         Index(i, j + y_dir_increment, k + z_dir_increment),
                                         Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                         Index(i + x_dir_increment, j, k + z_dir_increment), surface_dir);
                break;
            default:
                cerr << "wrong interpolation direction" << endl;
                exit(1);
        }
    }
    else if (argmax_axis == 2) {
        assert(z_dir_increment != 0);
        assert(isValidRange(i, j, k + z_dir_increment));
        auto next_coord = Intersect(vec, cur_coord, coord[Index(i, j, k + z_dir_increment)], PlaneDirection::XY_POS);

        IdxType x_changed = i + x_dir_increment;
        IdxType y_changed = j + y_dir_increment;
        // then determine if the ray cross other planes instead of X := x+x_dir*resolution
        assert(isValidRange(x_changed, j, k));
        Vec3 determiner_coord_for_x = coord[Index(x_changed, j, k)];
        assert(isValidRange(i, y_changed, k));
        Vec3 determiner_coord_for_z = coord[Index(i, y_changed, k)];
        // then determine if the ray firstly intersect the Y = a surface or Z = b surface before intersecting with X = c. a, b, c
        // are some constants
        Vec3 next_coord_cand1(next_coord);
        Vec3 next_coord_cand2(next_coord);
        bool flag_cand1 = false, flag_cand2 = false;
        if ((x_dir_increment > 0 && next_coord[0] > determiner_coord_for_x[0])
            || (x_dir_increment < 0 && next_coord[0] < determiner_coord_for_x[0])) {
            next_coord_cand1 = Intersect(vec, cur_coord, determiner_coord_for_x, PlaneDirection::YZ_POS);
            flag_cand1 = true;
        }
        if ((y_dir_increment > 0 && next_coord[1] > determiner_coord_for_z[1])
            || (y_dir_increment < 0 && next_coord[1] < determiner_coord_for_z[1])) {
            next_coord_cand2 = Intersect(vec, cur_coord, determiner_coord_for_z, PlaneDirection::XZ_POS);
            flag_cand2 = true;
        }
        // determine the increment of the index of four surfaces points
        int surface_dir = PlaneDirection::XY_POS;
        if ((z_dir_increment > 0 ? next_coord_cand1[2] < next_coord_cand2[2] : next_coord_cand1[2] > next_coord_cand2[2])
            && flag_cand1) {
            next_coord = next_coord_cand1;
            surface_dir = YZ_POS;
        }
        else if ((z_dir_increment > 0 ? next_coord_cand2[2] < next_coord_cand1[2] : next_coord_cand2[2] > next_coord_cand1[2])
                 && flag_cand2) {
            next_coord = next_coord_cand2;
            surface_dir = XZ_POS;
        }
        switch (surface_dir) {
            case YZ_POS:
                assert(isValidRange(i + x_dir_increment, j, k) && isValidRange(i + x_dir_increment, j + y_dir_increment, k)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i + x_dir_increment, j, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                         Index(i + x_dir_increment, j, k + z_dir_increment), surface_dir);
                break;
            case XZ_POS:
                assert(isValidRange(i, j + y_dir_increment, k) && isValidRange(i + x_dir_increment, j + y_dir_increment, k)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i, j + y_dir_increment, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i, j + y_dir_increment, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k),
                                         Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                         Index(i, j + y_dir_increment, k + z_dir_increment), surface_dir);
                break;
            case XY_POS:
                assert(isValidRange(i, j, k + z_dir_increment) && isValidRange(i, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment)
                       && isValidRange(i + x_dir_increment, j, k + z_dir_increment));
                temp_val = Interpolation(next_coord, Index(i, j, k + z_dir_increment),
                                         Index(i, j + y_dir_increment, k + z_dir_increment),
                                         Index(i + x_dir_increment, j + y_dir_increment, k + z_dir_increment),
                                         Index(i + x_dir_increment, j, k + z_dir_increment), surface_dir);
                break;
            default:
                cerr << "wrong interpolation direction" << endl;
                exit(1);
        }
    }
    else {
            cerr << "wrong argmax axis value" << endl;
            exit(1);
    }

    minval = min(temp_val, minval);
    return minval;
}

Vec3 Visibility::Intersect(const Vec3 &dir_vec, const Vec3 &cur_pt, const Vec3 &plane_pt, int direction) {
    Vec3 vec_prime = plane_pt - cur_pt;
    Vec3 normal;
    if (direction == PlaneDirection::XZ_POS || direction == PlaneDirection::XZ_NEG) {
        normal[1] = 1; normal[0] = normal[2] = 0;
    }
    else if (direction == PlaneDirection::YZ_POS || direction == PlaneDirection::YZ_NEG) {
        normal[0] = 1; normal[1] = normal[2] = 0;
    }
    else if (direction == PlaneDirection::XY_POS || direction == PlaneDirection::XY_NEG) {
        normal[2] = 1; normal[0] = normal[1] = 0;
    }
    else {
        cerr << "wrong direction passed" << endl;
        exit(1);
    }
    return cur_pt + dir_vec * (normal.dot(vec_prime)) / normal.dot(dir_vec);
}

dtype Visibility::Interpolation(const Vec3 &intersect_point, const unsigned long &sur_p1_idx, const unsigned long &sur_p2_idx,
                          const unsigned long &sur_p3_idx, const unsigned long &sur_p4_idx, int direction)
{
    dtype p1 = psi[sur_p1_idx];
    dtype p2 = psi[sur_p2_idx];
    dtype p3 = psi[sur_p3_idx];
    dtype p4 = psi[sur_p4_idx];
    const Vec3 &p1_coord = coord[sur_p1_idx];
    const Vec3 &p3_coord = coord[sur_p3_idx];
    // for the point whose neighbors are all positive or negative, sign matters.
    // So just take the average value;
//    if ((p1 > 0 && p2 > 0 && p3 > 0 && p4 > 0) || (p1 < 0 && p2 < 0 && p3 < 0 && p4 < 0)) {
//        return (p1 + p2 + p3 + p4) / 4;
//    }
    dtype val, denominator;
    //// bug fixed: p3 * xxx + p4 * xxx. original one(e.g) is p3 * (p3_coord[0] - intersect_point[0]) + p4 * (intersect_point[0] - p1_coord[0])
    if (direction == PlaneDirection::XZ_POS || direction == PlaneDirection::XZ_NEG) {
        assert(p1_coord[0] != p3_coord[0] && p1_coord[2] != p3_coord[2]);
        denominator = (p3_coord[2] - p1_coord[2]) * (p3_coord[0] - p1_coord[0]);
        val = ((p3_coord[2] - intersect_point[2]) * (p1 * (p3_coord[0] - intersect_point[0]) + p2 * (intersect_point[0] - p1_coord[0]))
               + (intersect_point[2] - p1_coord[2]) * (p3 * (intersect_point[0] - p1_coord[0]) + p4 * (p3_coord[0] - intersect_point[0])))
              / denominator;
    }
    else if (direction == PlaneDirection::YZ_POS || direction == PlaneDirection::YZ_NEG) {
        assert(p1_coord[1] != p3_coord[1] && p1_coord[2] != p3_coord[2]);
        denominator = (p3_coord[2] - p1_coord[2]) * (p3_coord[1] - p1_coord[1]);
        val = ((p3_coord[2] - intersect_point[2]) * (p1 * (p3_coord[1] - intersect_point[1]) + p2 * (intersect_point[1] - p1_coord[1]))
               + (intersect_point[2] - p1_coord[2]) * (p3 * (intersect_point[1] - p1_coord[1]) + p4 * (p3_coord[1] - intersect_point[1])))
              / denominator;
    }
    else if (direction == PlaneDirection::XY_POS || direction == PlaneDirection::XY_NEG) {
        assert(p1_coord[1] != p3_coord[1] && p1_coord[0] != p3_coord[0]);
        denominator = (p3_coord[1] - p1_coord[1]) * (p3_coord[0] - p1_coord[0]);
        val = ((p3_coord[1] - intersect_point[1]) * (p1 * (p3_coord[0] - intersect_point[0]) + p2 * (intersect_point[0] - p1_coord[0]))
               + (intersect_point[1] - p1_coord[1]) * (p3 * (intersect_point[0] - p1_coord[0]) + p4 * (p3_coord[0] - intersect_point[0])))
              / denominator;
    }
    else {
        cerr << "wrong direction in interpolation function" << endl;
        exit(1);
    }
    return val;
}





