//
// Created by himalaya on 10/24/19.
//

#include "grid3d.h"
#include <omp.h>
#include <queue>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

bool PointKeyVal::operator<(const struct PointKeyVal &rhs) const
{
    return this->phi_val > rhs.phi_val;
}


Grid3d::Grid3d(DimUnit height, DimUnit width, DimUnit depth):
        _height(height), _width(width), _depth(depth),
        active_distance(4.1), landmine_distance(5.9), boundary_distance(7.7)
{
    unsigned long total_num = height * width * depth;
    this->grid_prop.resize(total_num);
    this->phi.resize(total_num, INF);
//    this->velocity.resize(total_num, 0);
    this->Phi.resize(total_num, 0);
    this->narrow_band.resize(height);
    for (auto &iter : this->narrow_band) {iter.resize(width);}
    this->band_begin_i = height;
    this->band_end_i = 0;
    this->band_begin_j.resize(height, width);
    this->band_end_j.resize(height, 0);
    this->coord.resize(total_num);
//    this->velocity2.resize(total_num, 0);
//    this->d_Phi.resize(total_num, 0);
}

bool Grid3d::isValidRange(IdxType i, IdxType j, IdxType k) const
{
    return (i >= 0 && i < this->_height && j >= 0 && j < this->_width
            && k >= 0 && k < this->_depth);
}

void Grid3d::Build_band()
{
//#pragma omp parallel for default(none)
    for (auto i = this->band_begin_i; i < this->band_end_i; i++) {
        for (auto j = this->band_begin_j[i]; j < this->band_end_j[i]; j++) {
            // different from 2d case, we need to consider if there is element in between
            if (!this->narrow_band[i][j].empty()) {
                bool flag_interior = false;
                IdxType start = 0, end = 0;
                // the <= is crucial since the flag_interior is changed only when iterated outside the band.
                // bug fixed. no need for <=. In added another condition
                for (auto k = this->narrow_band[i][j].front().start; k < this->narrow_band[i][j].front().end; k++) {
                    if (!flag_interior) {
                        start = k;
                        end = k;
                    }
                    else { end = k; }
                    assert(isValidRange(i, j, k));
                    flag_interior = this->grid_prop[this->Index(i, j, k)].nb_status != NarrowBandStatus::OUTSIDE;
                    if (start != end && !flag_interior) {
                        this->narrow_band[i][j].emplace_back(NarrowBandExtent{start, end});
                    }
                    else if (k == this->narrow_band[i][j].front().end - 1 && flag_interior) {
                        // todo: wrong here
                        this->narrow_band[i][j].emplace_back(NarrowBandExtent{start, IdxType(k + 1)});
                    }
                }
                // we need to pop the element recording the coarse narrow band
                // different from 2d case
                this->narrow_band[i][j].pop_front();
            }
        }
        assert(this->band_begin_j[i] <= this->band_end_j[i]);
    }
    assert(this->band_begin_i <= this->band_end_i);
}

void Grid3d::Build_coarse_band(IdxType i, IdxType j, IdxType k)
{
    // end is one more step of the last element
    this->band_begin_i = i < this->band_begin_i ? i : this->band_begin_i;
    this->band_end_i = i > this->band_end_i - 1 ? i + 1 : this->band_end_i;
    this->band_begin_j[i] = j < this->band_begin_j[i] ? j : this->band_begin_j[i];
    this->band_end_j[i] = j > this->band_end_j[i] - 1 ? j + 1 : this->band_end_j[i];
    if (!this->narrow_band[i][j].empty()) {
        auto &start = this->narrow_band[i][j].front().start;
        auto &end = this->narrow_band[i][j].front().end;
        start = k < start ? k : start;
        end = k > end - 1 ? k + 1 : end;
    }
    else {
        this->narrow_band[i][j].emplace_back(NarrowBandExtent{k, (IdxType)(k + 1)});
    }
}

bool Grid3d::isFrontHere(IdxType i, IdxType j, IdxType k, std::vector<cv::Vec3i> &front_dir) const
{
//    unsigned short res = 0;
//    auto current_val = this->phi[this->Index(i, j, k)];
//    res = (current_val * this->phi[this->Index(i - 1, j, k)] <= 0) |
//            (current_val * this->phi[this->Index(i, j - 1, k)] <= 0) << 1u |
//            (current_val * this->phi[this->Index(i + 1, j, k)] <= 0) << 2u |
//            (current_val * this->phi[this->Index(i, j, k - 1)] <= 0) << 3u |
//            (current_val * this->phi[this->Index(i, j + 1, k)] <= 0) << 4u |
//            (current_val * this->phi[this->Index(i, j, k + 1)] <= 0) << 5u;

    bool is_front_here = false;
    front_dir.clear();
    assert(isValidRange(i, j, k));
    auto current_val = this->phi[this->Index(i, j, k)];
    if (current_val == 0)
        return true;
    int index[6][3] = {{-1, 0, 0}, {1, 0, 0},
                       {0, -1, 0}, {0, 1, 0},
                       {0, 0, -1}, {0, 0, 1}};
    for (auto &iter : index) {
        if (!isValidRange(i + iter[0], j + iter[1], k + iter[2])) { continue; }
        assert(isValidRange(i + iter[0], j + iter[1], k + iter[2]));
        if (current_val * this->phi[this->Index(i + iter[0], j + iter[1], k + iter[2])] <= 0) {
            front_dir.emplace_back(cv::Vec3i(iter[0], iter[1], iter[2]));
            is_front_here = true;
        }
    }
    return is_front_here;
}

bool Grid3d::isFrontHere(IdxType i, IdxType j, IdxType k, std::vector<cv::Vec3i> &front_dir, std::vector<cv::Vec3i> &aux_front_dir) const
{
//    unsigned short res = 0;
//    auto current_val = this->phi[this->Index(i, j, k)];
//    res = (current_val * this->phi[this->Index(i - 1, j, k)] <= 0) |
//            (current_val * this->phi[this->Index(i, j - 1, k)] <= 0) << 1u |
//            (current_val * this->phi[this->Index(i + 1, j, k)] <= 0) << 2u |
//            (current_val * this->phi[this->Index(i, j, k - 1)] <= 0) << 3u |
//            (current_val * this->phi[this->Index(i, j + 1, k)] <= 0) << 4u |
//            (current_val * this->phi[this->Index(i, j, k + 1)] <= 0) << 5u;

    bool is_front_here = false;
    front_dir.clear();
    assert(isValidRange(i, j, k));
    auto current_val = this->phi[this->Index(i, j, k)];
    int index[6][3] = {{-1, 0, 0}, {1, 0, 0},
                       {0, -1, 0}, {0, 1, 0},
                       {0, 0, -1}, {0, 0, 1}};
    for (auto &iter : index) {
        if (!isValidRange(i + iter[0], j + iter[1], k + iter[2])) { continue; }
        assert(isValidRange(i + iter[0], j + iter[1], k + iter[2]));
        auto exam_val = current_val * this->phi[this->Index(i + iter[0], j + iter[1], k + iter[2])];
        if (exam_val <= 0) {
            if (exam_val == 0)
                aux_front_dir.emplace_back(iter[0], iter[1], iter[2]);
            front_dir.emplace_back(cv::Vec3i(iter[0], iter[1], iter[2]));
            is_front_here = true;
        }
    }
    return is_front_here;
}

bool Grid3d::isRealFront(IdxType i, IdxType j, IdxType k, const std::vector<cv::Vec3i> &front_dir) const
{
    for (const auto &iter : front_dir) {
      if (phi[Index(i + iter[0], j + iter[1], k + iter[2])] != 0)
          return true;
    }
    return false;
}

void Grid3d::Marching_pos(std::priority_queue<PointKeyVal> &close_set)
{
    dtype speed = 1.0;
    dtype reciprocal = 1.0 / speed;
    int index[6][3] = {{-1, 0, 0}, {1, 0, 0},
                       {0, -1, 0}, {0, 1, 0},
                       {0, 0, -1}, {0, 0, 1}};
    while (!close_set.empty()) {
        PointKeyVal point = close_set.top();
        assert(isValidRange(point.i, point.j, point.k));
        if (point.phi_val <= active_distance) {
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::ACTIVE;
        }
        else if (point.phi_val <= landmine_distance) {
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::LANDMINE;
        }
        else if (point.phi_val <= boundary_distance) {
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::BOUNDARY;
        }
        else {
            // it means all the points remained are outside narrow band. So the loop can break.
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::OUTSIDE;
            break;
        }

        // add new 'CLOSE' point from 'FAR' point and computing the value
        for (const auto &idx_iter : index) {
            IdxType i = point.i + idx_iter[0];
            IdxType j = point.j + idx_iter[1];
            IdxType k = point.k + idx_iter[2];
            auto idx_ijk = this->Index(i, j, k);
            // check necessity of the grid to be computed
            if (this->isValidRange(i, j, k) && this->grid_prop[idx_ijk].fmm_status != FMM_Status::ACCEPT
                                            && this->grid_prop[idx_ijk].fmm_status != FMM_Status::OTHER_SIDE) {
                // smaller value in x, y, z direction respectively
                vector<dtype> smaller_val(3, 0);
                smaller_val[0] = min(this->isValidRange(i - 1, j, k) ? this->phi[this->Index(i - 1, j, k)] : INF,
                                          this->isValidRange(i + 1, j, k) ? this->phi[this->Index(i + 1, j, k)] : INF);
                smaller_val[1] = min(this->isValidRange(i, j - 1, k) ? this->phi[this->Index(i, j - 1, k)] : INF,
                                          this->isValidRange(i, j + 1, k) ? this->phi[this->Index(i, j + 1, k)] : INF);
                smaller_val[2] = min(this->isValidRange(i, j, k - 1) ? this->phi[this->Index(i, j, k - 1)] : INF,
                                          this->isValidRange(i, j, k + 1) ? this->phi[this->Index(i, j, k + 1)] : INF);
                // ascending order
                // so 'c' is the largest
                std::sort(smaller_val.begin(), smaller_val.end());
                auto a = smaller_val[0];
                auto b = smaller_val[1];
                auto c = smaller_val[2];
                dtype temp;
                // case 1
                // todo: (16, 16, 8) k = -1  a = 1 b = inf c = inf 理论上来说应该是会归入此类
                if (pow(c - a, 2) + pow(c - b, 2) < pow(reciprocal, 2)) {
                    temp = (a + b + c + sqrt(3 * pow(reciprocal, 2) - pow(c - b, 2) - pow(c - a, 2)
                                             - pow(b - a, 2))) / 3;
                }
                else if (pow(c - a, 2) + pow(c - b, 2) >= pow(reciprocal, 2)
                         && abs(b - a) < reciprocal) {
                    temp = (a + b + sqrt(2 * pow(reciprocal, 2) - pow(a - b, 2))) / 2;

                }
                else {
                    temp = a + reciprocal;
                }
                // store the minimum value
                this->phi[idx_ijk] = min(this->phi[idx_ijk], temp);
//                if (isnan(this->phi[idx_ijk])) {
//                    cerr << "nan" << endl;
//                    exit(1);
//                }
                // FMM guarantee that the newly inserted value is larger than heap top element.
                // this condition is in case of duplication
                if (this->grid_prop[idx_ijk].fmm_status != FMM_Status::CLOSE) {
                    close_set.emplace(PointKeyVal{i, j, k, this->phi[idx_ijk]});
                    this->grid_prop[idx_ijk].fmm_status = FMM_Status::CLOSE;
                    this->grid_prop[idx_ijk].extension_status = ExtensionStatus::EXTENSION;
                }
            }
        }
        assert(isValidRange(point.i, point.j, point.k));
        this->grid_prop[this->Index(point.i, point.j, point.k)].fmm_status = FMM_Status::ACCEPT;
        // todo:conside put it into 6 lines above
        if (this->grid_prop[this->Index(point.i, point.j, point.k)].extension_status == ExtensionStatus::EXTENSION)
            this->pos_marching_sequence.emplace_back(IndexSet{point.i, point.j, point.k});
        this->Build_coarse_band(point.i, point.j, point.k);
        // add current point into neg narrow band (inside)
        close_set.pop();
    }
}

void Grid3d::Marching_neg(std::priority_queue<PointKeyVal> &close_set)
{
    dtype speed = 1.0;
    dtype reciprocal = 1.0 / speed;
    int index[6][3] = {{-1, 0, 0}, {1, 0, 0},
                       {0, -1, 0}, {0, 1, 0},
                       {0, 0, -1}, {0, 0, 1}};
    while (!close_set.empty()) {
        PointKeyVal point = close_set.top();
        assert(isValidRange(point.i, point.j, point.k));
        if (point.phi_val <= active_distance) {
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::ACTIVE;
        }
        else if (point.phi_val <= landmine_distance) {
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::LANDMINE;
        }
        else if (point.phi_val <= boundary_distance) {
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::BOUNDARY;
        }
        else {
            // it means all the points remained are outside narrow band. So the loop can break.
            this->grid_prop[this->Index(point.i, point.j, point.k)].nb_status = NarrowBandStatus::OUTSIDE;
            break;
        }

        // add new 'CLOSE' point from 'FAR' point and computing the value
        for (const auto &idx_iter : index) {
            IdxType i = point.i + idx_iter[0];
            IdxType j = point.j + idx_iter[1];
            IdxType k = point.k + idx_iter[2];
            auto idx_ijk = this->Index(i, j, k);
            // check necessity of the grid to be computed
            if (this->isValidRange(i, j, k) && this->grid_prop[idx_ijk].fmm_status != FMM_Status::ACCEPT
                && this->grid_prop[idx_ijk].fmm_status != FMM_Status::OTHER_SIDE) {
                // smaller value in x, y, z direction respectively
                vector<dtype> smaller_val(3, 0);
                smaller_val[0] = min(this->isValidRange(i - 1, j, k) ? this->phi[this->Index(i - 1, j, k)] : INF,
                                     this->isValidRange(i + 1, j, k) ? this->phi[this->Index(i + 1, j, k)] : INF);
                smaller_val[1] = min(this->isValidRange(i, j - 1, k) ? this->phi[this->Index(i, j - 1, k)] : INF,
                                     this->isValidRange(i, j + 1, k) ? this->phi[this->Index(i, j + 1, k)] : INF);
                smaller_val[2] = min(this->isValidRange(i, j, k - 1) ? this->phi[this->Index(i, j, k - 1)] : INF,
                                     this->isValidRange(i, j, k + 1) ? this->phi[this->Index(i, j, k + 1)] : INF);
                // ascending order
                // so 'c' is the largest
                std::sort(smaller_val.begin(), smaller_val.end());
                auto a = smaller_val[0];
                auto b = smaller_val[1];
                auto c = smaller_val[2];
                dtype temp;
                // case 1
                // todo: (16, 16, 8) k = -1  a = 1 b = inf c = inf 理论上来说应该是会归入此类
                if (pow(c - a, 2) + pow(c - b, 2) < pow(reciprocal, 2)) {
                    temp = (a + b + c + sqrt(3 * pow(reciprocal, 2) - pow(c - b, 2) - pow(c - a, 2)
                                             - pow(b - a, 2))) / 3;
                }
                else if (pow(c - a, 2) + pow(c - b, 2) >= pow(reciprocal, 2)
                         && abs(b - a) < reciprocal) {
                    temp = (a + b + sqrt(2 * pow(reciprocal, 2) - pow(a - b, 2))) / 2;

                }
                else {
                    temp = a + reciprocal;
                }
                // store the minimum value
                this->phi[idx_ijk] = min(this->phi[idx_ijk], temp);
//                if (isnan(this->phi[idx_ijk])) {
//                    cerr << "nan" << endl;
//                    exit(1);
//                }
                // FMM guarantee that the newly inserted value is larger than heap top element.
                // this condition is in case of duplication
                if (this->grid_prop[idx_ijk].fmm_status != FMM_Status::CLOSE) {
                    close_set.emplace(PointKeyVal{i, j, k, this->phi[idx_ijk]});
                    this->grid_prop[idx_ijk].fmm_status = FMM_Status::CLOSE;
                    this->grid_prop[idx_ijk].extension_status = ExtensionStatus::EXTENSION;
                }
            }
        }
        assert(isValidRange(point.i, point.j, point.k));
        this->grid_prop[this->Index(point.i, point.j, point.k)].fmm_status = FMM_Status::ACCEPT;
        // todo:conside put it into 6 lines above
        if (this->grid_prop[this->Index(point.i, point.j, point.k)].extension_status == ExtensionStatus::EXTENSION)
            this->neg_marching_sequence.emplace_back(IndexSet{point.i, point.j, point.k});
        this->Build_coarse_band(point.i, point.j, point.k);
        // add current point into neg narrow band (inside)
        close_set.pop();
    }
    for (const auto &iter : neg_front_sequence) {
        assert(isValidRange(iter.i, iter.j, iter.k));
        this->phi[this->Index(iter.i, iter.j, iter.k)] *= -1;
    }
    for (const auto &iter : neg_marching_sequence) {
        assert(isValidRange(iter.i, iter.j, iter.k));
        this->phi[this->Index(iter.i, iter.j, iter.k)] *= -1;
    }
}

/*
void Grid3d::Extend_velocity()
{
    for (unsigned long idx = 0; idx < this->marching_sequence.size(); idx++) {
        auto &iter = this->marching_sequence[idx];
        IdxType i = iter.i;
        IdxType j = iter.j;
        IdxType k = iter.k;
        // todo: delete val after debug finished
        dtype val;
        assert(isValidRange(i, j, k));
        auto idx_ijk = this->Index(i, j, k);
        // first complete natural speed part
        if (this->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY
               && this->grid_prop[idx_ijk].extension_status == ExtensionStatus::NATURAL) {
            // central difference
            assert(isValidRange(i + 1, j, k) && isValidRange(i - 1, j, k));
            dtype phi_x = (this->phi[this->Index(i + 1, j, k)] - this->phi[this->Index(i - 1, j, k)]) / 2;
            assert(isValidRange(i, j + 1, k) && isValidRange(i, j - 1, k));
            dtype phi_y = (this->phi[this->Index(i, j + 1, k)] - this->phi[this->Index(i, j - 1, k)]) / 2;
            assert(isValidRange(i, j, k + 1) && isValidRange(i, j, k - 1));
            dtype phi_z = (this->phi[this->Index(i, j, k + 1)] - this->phi[this->Index(i, j, k - 1)]) / 2;
            if (phi_x !=0 || phi_y != 0 || phi_z != 0) {
                dtype phi_xx = this->phi[this->Index(i + 1, j, k)] + this->phi[this->Index(i - 1, j, k)] - 2 * this->phi[idx_ijk];
                dtype phi_yy = this->phi[this->Index(i, j + 1, k)] + this->phi[this->Index(i, j - 1, k)] - 2 * this->phi[idx_ijk];
                dtype phi_zz = this->phi[this->Index(i, j, k + 1)] + this->phi[this->Index(i, j, k - 1)] - 2 * this->phi[idx_ijk];
                assert(isValidRange(i + 1, j + 1, k) && isValidRange(i + 1, j - 1, k) && isValidRange(i - 1, j + 1, k) && isValidRange(i - 1, j - 1, k));
                dtype phi_xy = (this->phi[this->Index(i + 1, j + 1, k)] - this->phi[this->Index(i - 1, j + 1, k)]
                                 - this->phi[this->Index(i + 1, j - 1, k)] + this->phi[this->Index(i - 1, j - 1, k)]) / 4;
                assert(isValidRange(i, j + 1, k + 1) && isValidRange(i, j + 1, k - 1) && isValidRange(i, j - 1, k + 1) && isValidRange(i, j - 1, k - 1));
                dtype phi_yz = (this->phi[this->Index(i, j + 1, k + 1)] - this->phi[this->Index(i, j - 1, k + 1)]
                                - this->phi[this->Index(i, j + 1, k - 1)] + this->phi[this->Index(i, j - 1, k - 1)]) / 4;
                assert(isValidRange(i + 1, j, k + 1) && isValidRange(i + 1, j, k - 1) && isValidRange(i - 1, j, k + 1) && isValidRange(i - 1, j, k - 1));
                dtype phi_xz = (this->phi[this->Index(i + 1, j, k + 1)] - this->phi[this->Index(i - 1, j, k + 1)]
                                - this->phi[this->Index(i + 1, j, k - 1)] + this->phi[this->Index(i - 1, j, k - 1)]) / 4;
                val = ((pow(phi_y, 2) + pow(phi_z, 2)) * phi_xx
                                            + (pow(phi_x, 2) + pow(phi_z, 2)) * phi_yy
                                            + (pow(phi_x, 2) + pow(phi_y, 2)) * phi_zz
                                            - 2 * phi_x * phi_y * phi_xy
                                            - 2 * phi_x * phi_z * phi_xz
                                            - 2 * phi_y * phi_z * phi_yz)
                                           / (pow(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z, 1.5));
                this->Phi[idx_ijk] = val;
            }
            else {
                this->Phi[idx_ijk] = 0;

            }
        }
        // extension speed
        else if (this->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY
                    && this->grid_prop[idx_ijk].extension_status == ExtensionStatus::EXTENSION) {
            // ensure upwind scheme
            if (this->phi[idx_ijk] > 0) {
                IdxType argmin_i, argmin_j, argmin_k;
                if (i - 1 >= 0 && i + 1 < this->_height) {
                    assert(isValidRange(i + 1, j, k) && isValidRange(i - 1, j, k));
                    argmin_i = this->phi[this->Index(i - 1, j, k)] <= this->phi[this->Index(i + 1, j, k)] ? i - 1 : i + 1;
                }
                else {
                    argmin_i = i - 1 < 0 ? i + 1 : i - 1; }
                if (j - 1 >= 0 && j + 1 < this->_width) {
                    assert(isValidRange(i, j + 1, k) && isValidRange(i, j - 1, k));
                    argmin_j = this->phi[this->Index(i, j - 1, k)] <= this->phi[this->Index(i, j + 1, k)] ? j - 1 : j + 1;
                }
                else {
                    argmin_j = j - 1 < 0 ? j + 1 : j - 1;}
                if (k - 1 >= 0 && k + 1 < this->_depth) {
                    assert(isValidRange(i, j, k + 1) && isValidRange(i, j, k - 1));
                    argmin_k = this->phi[this->Index(i, j, k - 1)] <= this->phi[this->Index(i, j, k + 1)] ? k - 1 : k + 1;
                }
                else {
                    argmin_k = k - 1 < 0 ? k + 1 : k - 1;}
                // determine if fmm is reduced to 2d case, so there will be one direction not satisfying upwind scheme
                assert(isValidRange(argmin_i, j, k));
                dtype val_x_dir = this->phi[idx_ijk] - this->phi[this->Index(argmin_i, j, k)];
                if (val_x_dir < 0) val_x_dir = 0;
                assert(isValidRange(i, argmin_j, k));
                dtype val_y_dir = this->phi[idx_ijk] - this->phi[this->Index(i, argmin_j, k)];
                if (val_y_dir < 0) val_y_dir = 0;
                assert(isValidRange(i, j, argmin_k));
                dtype val_z_dir = this->phi[idx_ijk] - this->phi[this->Index(i, j, argmin_k)];
                if (val_z_dir < 0) val_z_dir = 0;
                val = (this->Phi[this->Index(argmin_i, j, k)] * val_x_dir + this->Phi[this->Index(i, argmin_j, k)] * val_y_dir
                            + this->Phi[this->Index(i, j, argmin_k)] * val_z_dir) / (val_x_dir + val_y_dir + val_z_dir);
//                val = (this->velocity[this->Index(argmin_i, j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(argmin_i, j, k)])
//                                           + this->velocity[this->Index(i, argmin_j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, argmin_j, k)])
//                                           + this->velocity[this->Index(i, j, argmin_k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, j, argmin_k)]))
//                                          / (3 * this->phi[idx_ijk] - this->phi[this->Index(argmin_i, j, k)]
//                                             - this->phi[this->Index(i, argmin_j, k)] - this->phi[this->Index(i, j, argmin_k)]);
            }
            else {
                assert(this->phi[idx_ijk] < 0);
                assert(isValidRange(i + 1, j, k) && isValidRange(i - 1, j, k));
                IdxType argmax_i = this->phi[this->Index(i - 1, j, k)] >= this->phi[this->Index(i + 1, j, k)] ? i - 1 : i + 1;
                assert(isValidRange(i, j + 1, k) && isValidRange(i, j - 1, k));
                IdxType argmax_j = this->phi[this->Index(i, j - 1, k)] >= this->phi[this->Index(i, j + 1, k)] ? j - 1 : j + 1;
                assert(isValidRange(i, j, k + 1) && isValidRange(i, j, k - 1));
                IdxType argmax_k = this->phi[this->Index(i, j, k - 1)] >= this->phi[this->Index(i, j, k + 1)] ? k - 1 : k + 1;
                assert(isValidRange(argmax_i, j, k));
                dtype val_x_dir = this->phi[idx_ijk] - this->phi[this->Index(argmax_i, j, k)];
                if (val_x_dir > 0) val_x_dir = 0;
                assert(isValidRange(i, argmax_j, k));
                dtype val_y_dir = this->phi[idx_ijk] - this->phi[this->Index(i, argmax_j, k)];
                if (val_y_dir > 0) val_y_dir = 0;
                assert(isValidRange(i, j, argmax_k));
                dtype val_z_dir = this->phi[idx_ijk] - this->phi[this->Index(i, j, argmax_k)];
                if (val_z_dir > 0) val_z_dir = 0;
                val = (this->Phi[this->Index(argmax_i, j, k)] * val_x_dir + this->Phi[this->Index(i, argmax_j, k)] * val_y_dir
                       + this->Phi[this->Index(i, j, argmax_k)] * val_z_dir) / (val_x_dir + val_y_dir + val_z_dir);
//                val = (this->velocity[this->Index(argmax_i, j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(argmax_i, j, k)])
//                                           + this->velocity[this->Index(i, argmax_j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, argmax_j, k)])
//                                           + this->velocity[this->Index(i, j, argmax_k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, j, argmax_k)]))
//                                          / (3 * this->phi[idx_ijk] - this->phi[this->Index(argmax_i, j, k)]
//                                             - this->phi[this->Index(i, argmax_j, k)] - this->phi[this->Index(i, j, argmax_k)]);
            }
            this->Phi[idx_ijk] = val;
        }
        assert(!isnan(this->Phi[idx_ijk]));
    }
}
*/

void Grid3d::Extend_velocity(PhiCalculator *velocity_calculator)
{
    cout << "start extend velocity" << endl;
    //    auto start = chrono::high_resolution_clock::now();
    auto start = omp_get_wtime();
    if (!velocity_calculator) { return; }
#pragma omp parallel for default(none) shared(velocity_calculator)
    for (unsigned long idx = 0; idx < this->pos_front_sequence.size(); idx++) {
        auto &point = this->pos_front_sequence[idx];
        IdxType i = point.i;
        IdxType j = point.j;
        IdxType k = point.k;
//        assert(isValidRange(i, j, k));
        auto idx_ijk = this->Index(i, j, k);
//        assert(this->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY
//               && this->grid_prop[idx_ijk].extension_status == ExtensionStatus::NATURAL);
//        auto start = chrono::high_resolution_clock::now();
        this->Phi[idx_ijk] = velocity_calculator->Compute_discrepancy(i, j, k, true);
#if USE_SIL
//        assert(this->Phi[idx_ijk] <= 2 && this->Phi[idx_ijk] >= 0);
#endif
//        auto stop = chrono::high_resolution_clock::now();
//        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
//        cout << "velocity extension cost time: " << duration.count() << endl;
//        assert(this->Phi[idx_ijk] <= 2 && this->Phi[idx_ijk] >= 0);
    }
//#pragma omp barrier

#pragma omp parallel for default(none) shared(velocity_calculator)
    for (unsigned long idx = 0; idx < this->neg_front_sequence.size(); idx++) {
        auto &point = this->neg_front_sequence[idx];
        IdxType i = point.i;
        IdxType j = point.j;
        IdxType k = point.k;
//        assert(isValidRange(i, j, k));
        auto idx_ijk = this->Index(i, j, k);
//        assert(this->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY
//               && this->grid_prop[idx_ijk].extension_status == ExtensionStatus::NATURAL);
        vector<cv::Vec3i> front_dir;
        front_dir.reserve(6);
//        assert(isFrontHere(i, j, k, front_dir) != 0);
        isFrontHere(i, j, k, front_dir);
        Determine_velocity_negative(i, j, k, front_dir);
    }
    //  claimed in the paper
    //    vector<cv::Vec3i> front_dir;
    //    front_dir.reserve(6);
    //    this->Phi[idx] = Determine_velocity_negative(this->phi, i, j, k, front_dir);
    // this->Phi[neg]
    // todo: neg extension part
#pragma omp barrier
    for (unsigned long idx = 0; idx < this->neg_marching_sequence.size(); idx++) {
        auto &point = this->neg_marching_sequence[idx];
        IdxType i = point.i;
        IdxType j = point.j;
        IdxType k = point.k;
//        assert(isValidRange(i, j, k));
        auto idx_ijk = this->Index(i, j, k);
        IdxType argmin_i = abs(this->phi[this->Index(i - 1, j, k)]) <= abs(this->phi[this->Index(i + 1, j, k)]) ? i - 1 : i + 1;
//        assert(isValidRange(i, j + 1, k) && isValidRange(i, j - 1, k));
        IdxType argmin_j = abs(this->phi[this->Index(i, j - 1, k)]) <= abs(this->phi[this->Index(i, j + 1, k)]) ? j - 1 : j + 1;
//        assert(isValidRange(i, j, k + 1) && isValidRange(i, j, k - 1));
        IdxType argmin_k = abs(this->phi[this->Index(i, j, k - 1)]) <= abs(this->phi[this->Index(i, j, k + 1)]) ? k - 1 : k + 1;
//        assert(isValidRange(argmin_i, j, k));
        dtype val_x_dir = this->phi[idx_ijk] - this->phi[this->Index(argmin_i, j, k)];
        if (val_x_dir > 0) val_x_dir = 0;
//        assert(isValidRange(i, argmin_j, k));
        dtype val_y_dir = this->phi[idx_ijk] - this->phi[this->Index(i, argmin_j, k)];
        if (val_y_dir > 0) val_y_dir = 0;
//        assert(isValidRange(i, j, argmin_k));
        dtype val_z_dir = this->phi[idx_ijk] - this->phi[this->Index(i, j, argmin_k)];
        if (val_z_dir > 0) val_z_dir = 0;
        dtype val = (this->Phi[this->Index(argmin_i, j, k)] * val_x_dir + this->Phi[this->Index(i, argmin_j, k)] * val_y_dir
               + this->Phi[this->Index(i, j, argmin_k)] * val_z_dir) / (val_x_dir + val_y_dir + val_z_dir);
        Phi[idx_ijk] = val;
    }
    // todo: pos extension part
//#pragma omp parallel for default(none)
    for (unsigned long idx = 0; idx < pos_marching_sequence.size(); idx++) {
        IdxType argmin_i, argmin_j, argmin_k;
        auto &point = this->pos_marching_sequence[idx];
        IdxType i = point.i;
        IdxType j = point.j;
        IdxType k = point.k;
//        assert(isValidRange(i, j, k));
        auto idx_ijk = Index(i, j, k);
        if (i - 1 >= 0 && i + 1 < this->_height) {
//            assert(isValidRange(i + 1, j, k) && isValidRange(i - 1, j, k));
            argmin_i = this->phi[this->Index(i - 1, j, k)] <= this->phi[this->Index(i + 1, j, k)] ? i - 1 : i + 1;
        }
        else {
            argmin_i = i - 1 < 0 ? i + 1 : i - 1; }
        if (j - 1 >= 0 && j + 1 < this->_width) {
//            assert(isValidRange(i, j + 1, k) && isValidRange(i, j - 1, k));
            argmin_j = this->phi[this->Index(i, j - 1, k)] <= this->phi[this->Index(i, j + 1, k)] ? j - 1 : j + 1;
        }
        else {
            argmin_j = j - 1 < 0 ? j + 1 : j - 1;}
        if (k - 1 >= 0 && k + 1 < this->_depth) {
//            assert(isValidRange(i, j, k + 1) && isValidRange(i, j, k - 1));
            argmin_k = this->phi[this->Index(i, j, k - 1)] <= this->phi[this->Index(i, j, k + 1)] ? k - 1 : k + 1;
        }
        else {
            argmin_k = k - 1 < 0 ? k + 1 : k - 1;}
        // determine if fmm is reduced to 2d case, so there will be one direction not satisfying upwind scheme
//        assert(isValidRange(argmin_i, j, k));
        dtype val_x_dir = this->phi[idx_ijk] - this->phi[this->Index(argmin_i, j, k)];
        if (val_x_dir < 0) val_x_dir = 0;
//        assert(isValidRange(i, argmin_j, k));
        dtype val_y_dir = this->phi[idx_ijk] - this->phi[this->Index(i, argmin_j, k)];
        if (val_y_dir < 0) val_y_dir = 0;
//        assert(isValidRange(i, j, argmin_k));
        dtype val_z_dir = this->phi[idx_ijk] - this->phi[this->Index(i, j, argmin_k)];
        if (val_z_dir < 0) val_z_dir = 0;
        dtype val = (this->Phi[this->Index(argmin_i, j, k)] * val_x_dir + this->Phi[this->Index(i, argmin_j, k)] * val_y_dir
               + this->Phi[this->Index(i, j, argmin_k)] * val_z_dir) / (val_x_dir + val_y_dir + val_z_dir);
        Phi[idx_ijk] = val;
    }

    cout << "velocity extension costs time: " << omp_get_wtime() - start << endl;
//    cout << "finished extension" << endl;
}

void Grid3d::Update_velocity(PhiCalculator *velocity_calculator)
{
    cout << "Update velocity" << endl;
    FMM3d(this, false, velocity_calculator);
}

Grid3d* Grid3d::Reinitialize(PhiCalculator *velocity_calculator)
{
    cout << "Reinitialization" << endl;
    return FMM3d(this, true, velocity_calculator);
}

Grid3d* FMM3d(Grid3d *init_grid, bool reinit, PhiCalculator *velocity_calculator)
{
    dtype float_err = 1e-12;
    //// FMM
    ////// 1. FMM initial step: find front
    //// parallel programming can be applied here
    auto height = init_grid->_height;
    auto width = init_grid->_width;
    auto depth = init_grid->_depth;
    Grid3d *new_grid = new Grid3d(height, width, depth);
    // index of points where front lies inside the grid
    vector<IndexSet> pos_val_front_index, neg_val_front_index;
    // a heap sort data structure to store grid points to be processed
    priority_queue<PointKeyVal> close_pq_pos, close_pq_neg;
    for (IdxType i = init_grid->band_begin_i; i < init_grid->band_end_i; i++) {
        for (IdxType j = init_grid->band_begin_j[i]; j < init_grid->band_end_j[i]; j++) {
            for (auto const &iter : init_grid->narrow_band[i][j]) {
                for (IdxType k = iter.start; k < iter.end; k++) {
                    assert(init_grid->isValidRange(i, j, k));
                    auto idx_ijk = init_grid->Index(i, j, k);
                    // todo: change != boundary to == active
                    if (init_grid->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY) {
                        // in 3-d case, we need to exam 6 directions
                        vector<cv::Vec3i> front_dir;
                        front_dir.reserve(6);
                        if (abs(init_grid->phi[idx_ijk]) < float_err)
                            init_grid->phi[idx_ijk] = 0;
                        // if in all the front directions the opposite value is 0, then we cannot put it into front_sequence
                        // since it will be the extension one
                        if (init_grid->isFrontHere(i, j, k, front_dir)) {
                            //// Maybe, here performance can be improved by check whether abs(phi_val) < a small number.
                            // narrowband status, phival, velocity
                            // Here all stuff including determining value\ sign\ narrowband status can be integrated into one part
                            Determine_front_property(init_grid, new_grid, i, j, k, front_dir);
                            // put >0 and  <0 into different set so that two direction fmm can be done.
                            if (init_grid->phi[idx_ijk] > 0) {
                                close_pq_pos.emplace(PointKeyVal{i, j, k, new_grid->phi[idx_ijk]});
                                if (init_grid->isRealFront(i, j, k, front_dir)) {
                                    new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::OTHER_SIDE;
                                    new_grid->pos_front_sequence.emplace_back(IndexSet{i, j, k});
                                    new_grid->grid_prop[idx_ijk].extension_status = ExtensionStatus::NATURAL;
                                }
                                else {
                                    new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::CLOSE;
                                    new_grid->grid_prop[idx_ijk].extension_status = ExtensionStatus::EXTENSION;
                                }
                            }
                            else if (init_grid->phi[idx_ijk] < 0) {
                                close_pq_neg.emplace(PointKeyVal{i, j, k, new_grid->phi[idx_ijk]});
                                if (init_grid->isRealFront(i, j, k, front_dir)) {
                                    new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::OTHER_SIDE;
                                    new_grid->neg_front_sequence.emplace_back(IndexSet{i, j, k});
                                    new_grid->grid_prop[idx_ijk].extension_status = ExtensionStatus::NATURAL;
                                }
                                else {
                                    new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::CLOSE;
                                    new_grid->grid_prop[idx_ijk].extension_status = ExtensionStatus::EXTENSION;
                                }
                            }
                            else {
//                                close_pq_pos.emplace(PointKeyVal{i, j, k, new_grid->phi[idx_ijk]});
                                new_grid->pos_front_sequence.emplace_back(IndexSet{i, j, k});
                                new_grid->grid_prop[idx_ijk].extension_status = ExtensionStatus::NATURAL;
//                                new_grid->front.emplace_back(IndexSet{i, j, k});
                            }
//                            new_grid->front.emplace_back(IndexSet{i, j, k});
                        }
                    }
                }
            }
        }
    }

    ////// 2. Marching
    new_grid->Marching_pos(close_pq_pos);
    new_grid->Marching_neg(close_pq_neg);

    ///// post precessing
    new_grid->Build_band();
    new_grid->Extend_velocity(velocity_calculator);
    if (!reinit) {
        // according to Sethian's paper, we only use new velocity instead of newly initialized phi.
        std::swap(new_grid->Phi, init_grid->Phi);
        delete new_grid;
        return nullptr;
    }
    else {
        std::swap(new_grid->coord, init_grid->coord);
        return new_grid;
    }
}

// return the front type
//// todo : design a data structure to store special points' index or normal points' index
int Determine_front_type(const std::vector<cv::Vec3i> &front_dir, std::vector<int> &special_grid_index)
{
    cv::Vec3i sum(0, 0, 0);
    int num_ones = 0;
    for (const auto &iter : front_dir) sum += iter;
    // non-zeros position of sum indicates the special grid point also has non-zero value at this position.
    // so the initial size should be zero
    vector<int> non_zero_pos;
    // there are only x, y, z three entries.
    non_zero_pos.reserve(3);
    for (IdxType i = 0; i < 3; i++) {
        if (abs(sum[i]) == 1) {
            num_ones++;
            non_zero_pos.emplace_back(i);
        }
    }
    switch (front_dir.size()) {
        case 1 :
            return FrontType::TYPE_A;
        case 2:
            if (num_ones == 2)
                return TYPE_B1;
            else
                // num_ones == 0
                return TYPE_B2;
        case 3:
            if (num_ones == 3)
                return TYPE_C1;
            else
                // num_ones = 1
                assert(num_ones == 1);
                // todo: efficiency could be improved here. Maybe use hasp table and change the logic
                // there should be only 1 non_zero_pos
                assert(non_zero_pos.size() == 1);
                for (const auto &iter_1 : non_zero_pos) {
                    // 3 here is the size of front_dir
                    for (int i = 0; i < 3; i++) {
                        if (abs(front_dir[i][iter_1]) == 1)
                            special_grid_index.emplace_back(i);
                    }
                }
                return TYPE_C2;
        case 4:
            if (num_ones == 0)
                return TYPE_D2;
            else {
                // num_ones == 2
                assert(num_ones == 2);
                assert(non_zero_pos.size() == 2);
                for (const auto &iter_1 : non_zero_pos) {
                    // 4 here is the size of front_dir
                    for (int i = 0; i < 4; i++) {
                        if (abs(front_dir[i][iter_1]) == 1)
                            special_grid_index.emplace_back(i);
                    }
                }
            }
            return TYPE_D1;
        case 5:
            assert(num_ones == 1);
            assert(non_zero_pos.size() == 1);
            for (const auto &iter_1 : non_zero_pos) {
                // 5 here is the size of front_dir
                for (int i = 0; i < 5; i++) {
                    if (abs(front_dir[i][iter_1]) == 1)
                        special_grid_index.emplace_back(i);
                }
            }
            return TYPE_E;
        case 6:
            assert(num_ones == 0);
            return TYPE_F;
    }
}

// todo: efficiency can be improved by store pow(x, 2) into a variable
void Determine_front_property(Grid3d *old_grid, Grid3d *new_grid, IdxType i, IdxType j, IdxType k,
                              std::vector<cv::Vec3i> &front_dir)
{
    // first deal with phi = 0
    assert(old_grid->isValidRange(i, j, k));
    auto idx_ijk = old_grid->Index(i, j, k);
    if (old_grid->phi[idx_ijk] == 0) {
        new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::ACCEPT;
        new_grid->grid_prop[idx_ijk].nb_status = NarrowBandStatus::ACTIVE;
        new_grid->phi[idx_ijk] = 0;
        return;
    }
    vector<int> special_grid_index;
    special_grid_index.reserve(2);
    auto front_type = Determine_front_type(front_dir, special_grid_index);
    dtype dist1, dist2, dist3, dist4, dist5, dist6, temp_dist1, temp_dist2, temp_dist3;
    int s_index1, s_index2, r_index1, r_index2, r_index3, r_index4;
    switch (front_type) {
        // refer to head file for the definition of each type
        case TYPE_A:
            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1],  k + front_dir[0][2]));
            // use point-line distance for approximation
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk]
                                              - old_grid->phi[old_grid->Index(i + front_dir[0][0],
                                                                              j + front_dir[0][1],
                                                                              k + front_dir[0][2])]);
            new_grid->phi[idx_ijk] = dist1;
            break;
        case TYPE_B1:
            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
                    && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2]));
            // use point-line distance for approximation
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            new_grid->phi[idx_ijk] = sqrt(pow(dist1 * dist2, 2) / (pow(dist1, 2) + pow(dist2, 2)));
            break;
        case TYPE_B2:
            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
                    && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2]));
            // use point-line distance for approximation
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            new_grid->phi[idx_ijk] = min(dist1, dist2);
            break;
        case TYPE_C1:
            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
                    && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])
                    && old_grid->isValidRange(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2]));
            // precisely compute point-surface distance
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            dist3 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[2][0],
                                                                                                     j + front_dir[2][1],
                                                                                                     k + front_dir[2][2])]);
            new_grid->phi[idx_ijk] = sqrt(pow(dist1 * dist2 * dist3, 2) / (pow(dist1, 2) + pow(dist2, 2) + pow(dist3, 2)));
            break;
        case TYPE_C2:

            // use point-line distance for approximation.
            // In this case, we need to find the special grid point, which is the peak point of a triangle
            // special point index
            s_index1 = special_grid_index[0];
            // regular point index
            r_index1 = (s_index1 + 1) % 3;
            // ensure the result is positive
            r_index2 = ((s_index1 - 1) % 3 + 3) % 3;
            assert(old_grid->isValidRange(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])
                   && old_grid->isValidRange(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])
                   && old_grid->isValidRange(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2]));

            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[s_index1][0],
                                                                                                     j + front_dir[s_index1][1],
                                                                                                     k + front_dir[s_index1][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index1][0],
                                                                                                     j + front_dir[r_index1][1],
                                                                                                     k + front_dir[r_index1][2])]);
            dist3 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index2][0],
                                                                                                     j + front_dir[r_index2][1],
                                                                                                     k + front_dir[r_index2][2])]);
            temp_dist1 = min(dist2, dist3);
            new_grid->phi[idx_ijk] = sqrt(pow(dist1 * temp_dist1, 2) / (pow(dist1, 2) + pow(temp_dist1, 2)));
            break;
        case TYPE_D1:

            // precisely compute point-surface distance
            // In this case, we need to find two special grid points, which constitute common edge of two tetrahedrons
            // special points index
            s_index1 = special_grid_index[0];
            s_index2 = special_grid_index[1];
            // regular points' index
            r_index1 = (s_index1 + 2) % 4;
            r_index2 = (s_index2 + 2) % 4;
            assert(old_grid->isValidRange(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])
                   && old_grid->isValidRange(i + front_dir[s_index2][0], j + front_dir[s_index2][1], k + front_dir[s_index2][2])
                   && old_grid->isValidRange(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])
                   && old_grid->isValidRange(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2]));

            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[s_index1][0],
                                                                                                     j + front_dir[s_index1][1],
                                                                                                     k + front_dir[s_index1][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[s_index2][0],
                                                                                                     j + front_dir[s_index2][1],
                                                                                                     k + front_dir[s_index2][2])]);
            dist3 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index1][0],
                                                                                                     j + front_dir[r_index1][1],
                                                                                                     k + front_dir[r_index1][2])]);
            dist4 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index2][0],
                                                                                                     j + front_dir[r_index2][1],
                                                                                                     k + front_dir[r_index2][2])]);
            temp_dist1 = min(dist3, dist4);
            new_grid->phi[idx_ijk] = sqrt(pow(dist1 * dist2 * temp_dist1, 2) / (pow(dist1, 2) + pow(dist2, 2) + pow(temp_dist1, 2)));
            break;
        case TYPE_D2:
            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
                   && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])
                   && old_grid->isValidRange(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2])
                   && old_grid->isValidRange(i + front_dir[3][0], j + front_dir[3][1], k + front_dir[3][2]));
            // use point-line distance to approximate
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            dist3 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[2][0],
                                                                                                     j + front_dir[2][1],
                                                                                                     k + front_dir[2][2])]);
            dist4 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[3][0],
                                                                                                     j + front_dir[3][1],
                                                                                                     k + front_dir[3][2])]);
            temp_dist1 = min(dist1, dist2);
            temp_dist2 = min(dist3, dist4);
            new_grid->phi[idx_ijk] = sqrt(pow(temp_dist1 * temp_dist2, 2) / (pow(temp_dist1, 2) + pow(temp_dist2, 2)));
            break;
        case TYPE_E:
            // precisely compute point-surface distance
            s_index1 = special_grid_index[0];
            r_index1 = (s_index1 + 1) % 5;
            r_index2 = (s_index1 + 2) % 5;
            r_index3 = (s_index1 + 3) % 5;
            r_index4 = (s_index1 + 4) % 5;
            assert(old_grid->isValidRange(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])
                   && old_grid->isValidRange(i + front_dir[r_index3][0], j + front_dir[r_index3][1], k + front_dir[r_index3][2])
                   && old_grid->isValidRange(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])
                   && old_grid->isValidRange(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2])
                   && old_grid->isValidRange(i + front_dir[r_index4][0], j + front_dir[r_index4][1], k + front_dir[r_index4][2]));

            // dist1 is the peak point of the large tetrahedron
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[s_index1][0],
                                                                                                     j + front_dir[s_index1][1],
                                                                                                     k + front_dir[s_index1][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index1][0],
                                                                                                     j + front_dir[r_index1][1],
                                                                                                     k + front_dir[r_index1][2])]);
            dist3 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index2][0],
                                                                                                     j + front_dir[r_index2][1],
                                                                                                     k + front_dir[r_index2][2])]);
            dist4 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index3][0],
                                                                                                     j + front_dir[r_index3][1],
                                                                                                     k + front_dir[r_index3][2])]);
            dist5 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[r_index4][0],
                                                                                                     j + front_dir[r_index4][1],
                                                                                                     k + front_dir[r_index4][2])]);
            temp_dist1 = min(dist2, dist3);
            temp_dist2 = min(dist4, dist5);
            new_grid->phi[idx_ijk] = sqrt(pow(dist1 * temp_dist1 * temp_dist2, 2) /
                                                  (pow(dist1, 2) + pow(temp_dist1, 2) + pow(temp_dist2, 2)));
            break;
        case TYPE_F:
            // precisely compute point-surface distance
            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
                   && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])
                   && old_grid->isValidRange(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2])
                   && old_grid->isValidRange(i + front_dir[3][0], j + front_dir[3][1], k + front_dir[3][2])
                   && old_grid->isValidRange(i + front_dir[4][0], j + front_dir[4][1], k + front_dir[4][2])
                   && old_grid->isValidRange(i + front_dir[5][0], j + front_dir[5][1], k + front_dir[5][2]));
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            dist3 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[2][0],
                                                                                                     j + front_dir[2][1],
                                                                                                     k + front_dir[2][2])]);
            dist4 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[3][0],
                                                                                                     j + front_dir[3][1],
                                                                                                     k + front_dir[3][2])]);
            dist5 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[4][0],
                                                                                                     j + front_dir[4][1],
                                                                                                     k + front_dir[4][2])]);
            dist6 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk] - old_grid->phi[old_grid->Index(i + front_dir[5][0],
                                                                                                     j + front_dir[5][1],
                                                                                                     k + front_dir[5][2])]);
            temp_dist1 = min(dist1, dist2);
            temp_dist2 = min(dist3, dist4);
            temp_dist3 = min(dist5, dist6);
            new_grid->phi[idx_ijk] = sqrt(pow(temp_dist1 * temp_dist2 * temp_dist3, 2) /
                                                  (pow(temp_dist1, 2) + pow(temp_dist2, 2) + pow(temp_dist3, 2)));
            // deal with singular point (the value depending on band width) I chose threshold 1.8 here
            if (new_grid->phi[idx_ijk] > 1.8) new_grid->phi[idx_ijk] = 1;
            break;
        default:
            cerr << "Wrong front type!" << endl;
            exit(1);
    }
    assert (!isnan(new_grid->phi[idx_ijk]));
}

dtype Grid3d::Determine_velocity_negative(IdxType i, IdxType j, IdxType k, const std::vector<cv::Vec3i> &front_dir)
{
    auto idx_ijk = this->Index(i, j, k);
    vector<int> special_grid_index;
    special_grid_index.reserve(2);
    auto front_type = Determine_front_type(front_dir, special_grid_index);
    dtype dist1, dist2, dist3, dist4, dist5, dist6, temp_dist1, temp_dist2, temp_dist3;
    dtype Phi1, Phi2, Phi3, Phi4, Phi5, Phi6, temp_Phi1, temp_Phi2, temp_Phi3;
    int s_index1, s_index2, r_index1, r_index2, r_index3, r_index4;
    switch (front_type) {
        // refer to head file for the definition of each type
        case TYPE_A:
//            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1],  k + front_dir[0][2]));
            // use point-line distance for approximation
            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk]
                                              - this->phi[Index(i + front_dir[0][0],
                                                                              j + front_dir[0][1],
                                                                              k + front_dir[0][2])]);
            Phi[idx_ijk] = Phi[Index(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])];
            break;
        case TYPE_B1:
//            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
//                   && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2]));
            // use point-line distance for approximation
            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            Phi1 = Phi[Index(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])];
            Phi2 = Phi[Index(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])];
            Phi[idx_ijk] = (pow(dist1, 2) * Phi2 + pow(dist2, 2) * Phi1) / (pow(dist1, 2) + pow(dist2, 2));
            break;
        case TYPE_B2:
//            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
//                   && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2]));
            // use point-line distance for approximation
            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            Phi1 = Phi[Index(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])];
            Phi2 = Phi[Index(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])];
            Phi[idx_ijk] = dist1 < dist2 ? Phi1 : Phi2;
            break;
        case TYPE_C1:
//            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
//                   && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])
//                   && old_grid->isValidRange(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2]));
            // precisely compute point-surface distance
            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            dist3 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[2][0],
                                                                                                     j + front_dir[2][1],
                                                                                                     k + front_dir[2][2])]);
            Phi1 = Phi[Index(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])];
            Phi2 = Phi[Index(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])];
            Phi3 = Phi[Index(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2])];
            Phi[idx_ijk] = (pow(dist1 * dist2, 2)  * Phi3 + pow(dist1 * dist3, 2) * Phi2 + pow(dist2 * dist3, 2) * Phi1)
                            / (pow(dist1, 2) + pow(dist2, 2) + pow(dist3, 2));
            break;
        case TYPE_C2:

            // use point-line distance for approximation.
            // In this case, we need to find the special grid point, which is the peak point of a triangle
            // special point index
            s_index1 = special_grid_index[0];
            // regular point index
            r_index1 = (s_index1 + 1) % 3;
            // ensure the result is positive
            r_index2 = ((s_index1 - 1) % 3 + 3) % 3;
//            assert(old_grid->isValidRange(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])
//                   && old_grid->isValidRange(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])
//                   && old_grid->isValidRange(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2]));

            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[s_index1][0],
                                                                                                     j + front_dir[s_index1][1],
                                                                                                     k + front_dir[s_index1][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index1][0],
                                                                                                     j + front_dir[r_index1][1],
                                                                                                     k + front_dir[r_index1][2])]);
            dist3 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index2][0],
                                                                                                     j + front_dir[r_index2][1],
                                                                                                     k + front_dir[r_index2][2])]);
            Phi1 = Phi[Index(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])];
            Phi2 = Phi[Index(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])];
            Phi3 = Phi[Index(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2])];
            temp_dist1 = min(dist2, dist3);
            temp_Phi1 = dist2 < dist3 ? Phi2 : Phi3;
            Phi[idx_ijk] = (pow(dist1, 2) * temp_Phi1 + pow(temp_dist1, 2) * Phi1) / (pow(dist1, 2) + pow(temp_dist1, 2));
            break;
        case TYPE_D1:

            // precisely compute point-surface distance
            // In this case, we need to find two special grid points, which constitute common edge of two tetrahedrons
            // special points index
            s_index1 = special_grid_index[0];
            s_index2 = special_grid_index[1];
            // regular points' index
            r_index1 = (s_index1 + 2) % 4;
            r_index2 = (s_index2 + 2) % 4;
//            assert(old_grid->isValidRange(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])
//                   && old_grid->isValidRange(i + front_dir[s_index2][0], j + front_dir[s_index2][1], k + front_dir[s_index2][2])
//                   && old_grid->isValidRange(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])
//                   && old_grid->isValidRange(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2]));

            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[s_index1][0],
                                                                                                     j + front_dir[s_index1][1],
                                                                                                     k + front_dir[s_index1][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[s_index2][0],
                                                                                                     j + front_dir[s_index2][1],
                                                                                                     k + front_dir[s_index2][2])]);
            dist3 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index1][0],
                                                                                                     j + front_dir[r_index1][1],
                                                                                                     k + front_dir[r_index1][2])]);
            dist4 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index2][0],
                                                                                                     j + front_dir[r_index2][1],
                                                                                                     k + front_dir[r_index2][2])]);
            Phi1 = Phi[Index(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])];
            Phi2 = Phi[Index(i + front_dir[s_index2][0], j + front_dir[s_index2][1], k + front_dir[s_index2][2])];
            Phi3 = Phi[Index(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])];
            Phi4 = Phi[Index(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2])];
            temp_dist1 = min(dist3, dist4);
            temp_Phi1 = dist3 < dist4 ? Phi3 : Phi4;
            Phi[idx_ijk] = (pow(dist1 * dist2, 2)  * temp_Phi1 + pow(dist1 * temp_dist1, 2) * Phi2 + pow(dist2 * temp_dist1, 2) * Phi1)
                           / (pow(dist1, 2) + pow(dist2, 2) + pow(temp_dist1, 2));
            break;
        case TYPE_D2:
//            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
//                   && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])
//                   && old_grid->isValidRange(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2])
//                   && old_grid->isValidRange(i + front_dir[3][0], j + front_dir[3][1], k + front_dir[3][2]));
            // use point-line distance to approximate
            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            dist3 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[2][0],
                                                                                                     j + front_dir[2][1],
                                                                                                     k + front_dir[2][2])]);
            dist4 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[3][0],
                                                                                                     j + front_dir[3][1],
                                                                                                     k + front_dir[3][2])]);
            Phi1 = Phi[Index(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])];
            Phi2 = Phi[Index(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])];
            Phi3 = Phi[Index(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2])];
            Phi4 = Phi[Index(i + front_dir[3][0], j + front_dir[3][1], k + front_dir[3][2])];
            temp_dist1 = min(dist1, dist2);
            temp_dist2 = min(dist3, dist4);
            temp_Phi1 = dist1 < dist2 ? Phi1 : Phi2;
            temp_Phi2 = dist3 < dist4 ? Phi3 : Phi4;
            Phi[idx_ijk] = (pow(temp_dist1, 2) * temp_Phi2 + pow(temp_dist2, 2) * temp_Phi1) / (pow(temp_dist1, 2) + pow(temp_dist2, 2));
            break;
        case TYPE_E:
            // precisely compute point-surface distance
            s_index1 = special_grid_index[0];
            r_index1 = (s_index1 + 1) % 5;
            r_index2 = (s_index1 + 2) % 5;
            r_index3 = (s_index1 + 3) % 5;
            r_index4 = (s_index1 + 4) % 5;
//            assert(old_grid->isValidRange(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])
//                   && old_grid->isValidRange(i + front_dir[r_index3][0], j + front_dir[r_index3][1], k + front_dir[r_index3][2])
//                   && old_grid->isValidRange(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])
//                   && old_grid->isValidRange(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2])
//                   && old_grid->isValidRange(i + front_dir[r_index4][0], j + front_dir[r_index4][1], k + front_dir[r_index4][2]));

            // dist1 is the peak point of the large tetrahedron
            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[s_index1][0],
                                                                                                     j + front_dir[s_index1][1],
                                                                                                     k + front_dir[s_index1][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index1][0],
                                                                                                     j + front_dir[r_index1][1],
                                                                                                     k + front_dir[r_index1][2])]);
            dist3 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index2][0],
                                                                                                     j + front_dir[r_index2][1],
                                                                                                     k + front_dir[r_index2][2])]);
            dist4 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index3][0],
                                                                                                     j + front_dir[r_index3][1],
                                                                                                     k + front_dir[r_index3][2])]);
            dist5 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[r_index4][0],
                                                                                                     j + front_dir[r_index4][1],
                                                                                                     k + front_dir[r_index4][2])]);
            Phi1 = Phi[Index(i + front_dir[s_index1][0], j + front_dir[s_index1][1], k + front_dir[s_index1][2])];
            Phi2 = Phi[Index(i + front_dir[r_index1][0], j + front_dir[r_index1][1], k + front_dir[r_index1][2])];
            Phi3 = Phi[Index(i + front_dir[r_index2][0], j + front_dir[r_index2][1], k + front_dir[r_index2][2])];
            Phi4 = Phi[Index(i + front_dir[r_index3][0], j + front_dir[r_index3][1], k + front_dir[r_index3][2])];
            Phi5 = Phi[Index(i + front_dir[r_index4][0], j + front_dir[r_index4][1], k + front_dir[r_index4][2])];
            temp_dist1 = min(dist2, dist3);
            temp_dist2 = min(dist4, dist5);
            temp_Phi1 = dist2 < dist3 ? Phi2 : Phi3;
            temp_Phi2 = dist4 < dist5 ? Phi4 : Phi5;
            Phi[idx_ijk] = (pow(dist1 * temp_dist1, 2)  * temp_Phi2 + pow(dist1 * temp_dist2, 2) * temp_Phi1 + pow(temp_dist2 * temp_dist1, 2) * Phi1)
                           / (pow(dist1, 2) + pow(temp_dist2, 2) + pow(temp_dist1, 2));
            break;
        case TYPE_F:
            // precisely compute point-surface distance
//            assert(old_grid->isValidRange(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])
//                   && old_grid->isValidRange(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])
//                   && old_grid->isValidRange(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2])
//                   && old_grid->isValidRange(i + front_dir[3][0], j + front_dir[3][1], k + front_dir[3][2])
//                   && old_grid->isValidRange(i + front_dir[4][0], j + front_dir[4][1], k + front_dir[4][2])
//                   && old_grid->isValidRange(i + front_dir[5][0], j + front_dir[5][1], k + front_dir[5][2]));
            dist1 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[0][0],
                                                                                                     j + front_dir[0][1],
                                                                                                     k + front_dir[0][2])]);
            dist2 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[1][0],
                                                                                                     j + front_dir[1][1],
                                                                                                     k + front_dir[1][2])]);
            dist3 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[2][0],
                                                                                                     j + front_dir[2][1],
                                                                                                     k + front_dir[2][2])]);
            dist4 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[3][0],
                                                                                                     j + front_dir[3][1],
                                                                                                     k + front_dir[3][2])]);
            dist5 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[4][0],
                                                                                                     j + front_dir[4][1],
                                                                                                     k + front_dir[4][2])]);
            dist6 = this->phi[idx_ijk] / (this->phi[idx_ijk] - this->phi[Index(i + front_dir[5][0],
                                                                                                     j + front_dir[5][1],
                                                                                                     k + front_dir[5][2])]);
            Phi1 = Phi[Index(i + front_dir[0][0], j + front_dir[0][1], k + front_dir[0][2])];
            Phi2 = Phi[Index(i + front_dir[1][0], j + front_dir[1][1], k + front_dir[1][2])];
            Phi3 = Phi[Index(i + front_dir[2][0], j + front_dir[2][1], k + front_dir[2][2])];
            Phi4 = Phi[Index(i + front_dir[3][0], j + front_dir[3][1], k + front_dir[3][2])];
            Phi5 = Phi[Index(i + front_dir[4][0], j + front_dir[4][1], k + front_dir[4][2])];
            Phi6 = Phi[Index(i + front_dir[5][0], j + front_dir[5][1], k + front_dir[5][2])];
            temp_dist1 = min(dist1, dist2);
            temp_dist2 = min(dist3, dist4);
            temp_dist3 = min(dist5, dist6);
            temp_Phi1 = dist1 < dist2 ? Phi1 : Phi2;
            temp_Phi2 = dist3 < dist4 ? Phi3 : Phi4;
            temp_Phi3 = dist5 < dist6 ? Phi5 : Phi6;
            Phi[idx_ijk] = (pow(temp_dist2 * temp_dist1, 2)  * temp_Phi3 + pow(temp_dist3 * temp_dist2, 2) * temp_Phi1 + pow(temp_dist3 * temp_dist1, 2) * temp_Phi2)
                           / (pow(temp_dist3, 2) + pow(temp_dist2, 2) + pow(temp_dist1, 2));
            // deal with singular point (the value depending on band width) I chose threshold 1.8 here
//            if (new_grid->phi[idx_ijk] > 1.8) new_grid->phi[idx_ijk] = 1;
            break;
        default:
            cerr << "Wrong front type!" << endl;
            exit(1);
    }
    assert (!isnan(Phi[idx_ijk]));
    assert (Phi[idx_ijk] <= 2 && Phi[idx_ijk] >= 0);
    return 0;
}


