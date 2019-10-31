//
// Created by himalaya on 10/24/19.
//

#include "grid3d.h"
#include <omp.h>
#include <queue>
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
    this->velocity.resize(total_num);
    this->narrow_band.resize(height);
    for (auto &iter : this->narrow_band) {iter.resize(width);}
    this->band_begin_i = height;
    this->band_end_i = 0;
    this->band_begin_j.resize(height, width);
    this->band_end_j.resize(height, 0);
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
                for (auto k = this->narrow_band[i][j].front().start; k <= this->narrow_band[i][j].front().end; k++) {
                    if (!flag_interior) {
                        start = k;
                        end = k;
                    }
                    else { end = k; }
                    flag_interior = this->grid_prop[this->Index(i, j, k)].nb_status != NarrowBandStatus::OUTSIDE;
                    if (start != end && !flag_interior) {
                        this->narrow_band[i][j].emplace_back(NarrowBandExtent{start, end});
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
    auto current_val = this->phi[this->Index(i, j, k)];
    int index[6][3] = {{-1, 0, 0}, {1, 0, 0},
                       {0, -1, 0}, {0, 1, 0},
                       {0, 0, -1}, {0, 0, 1}};
    for (auto &iter : index) {
        if (current_val * this->phi[this->Index(i + iter[0], j + iter[1], k + iter[2])] <= 0) {
            front_dir.emplace_back(cv::Vec3i(iter[0], iter[1], iter[2]));
            is_front_here = true;
        }
    }
    return is_front_here;
}

void Grid3d::Marching(std::priority_queue<PointKeyVal> &close_set, bool inside)
{
    dtype speed = 1.0;
    dtype reciprocal = 1.0 / speed;
    int index[6][3] = {{-1, 0, 0}, {1, 0, 0},
                       {0, -1, 0}, {0, 1, 0},
                       {0, 0, -1}, {0, 0, 1}};
    vector<IndexSet> band_point_index;
    while (!close_set.empty()) {
        PointKeyVal point = close_set.top();
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

                // FMM guarantee that the newly inserted value is larger than heap top element.
                // this condition is in case of duplication
                if (this->grid_prop[idx_ijk].fmm_status != FMM_Status::CLOSE) {
                    close_set.emplace(PointKeyVal{i, j, k, this->phi[idx_ijk]});
                    this->grid_prop[idx_ijk].fmm_status = FMM_Status::CLOSE;
                }
                this->grid_prop[idx_ijk].extension_status = ExtensionStatus::EXTENSION;
            }
        }
        this->grid_prop[this->Index(point.i, point.j, point.k)].fmm_status = FMM_Status::ACCEPT;
        this->marching_sequence.emplace_back(IndexSet{point.i, point.j, point.k});
        this->Build_coarse_band(point.i, point.j, point.k);
        // add current point into neg narrow band (inside)
        if (inside) { band_point_index.emplace_back(IndexSet{point.i, point.j, point.k}); }
        close_set.pop();
    }
    if (inside) {
        for (const auto &iter : band_point_index) {
            this->phi[this->Index(iter.i, iter.j, iter.k)] *= -1;
        }
    }
}

void Grid3d::Extend_velocity()
{
//#pragma omp parallel for default (none)
    for (unsigned long idx = 0; idx < this->marching_sequence.size(); idx++) {
        auto &iter = this->marching_sequence[idx];
        IdxType i = iter.i;
        IdxType j = iter.j;
        IdxType k = iter.k;
        // todo: delete val after debug finished
        dtype val;
        auto idx_ijk = this->Index(i, j, k);
        // first complete natural speed part
        if (this->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY
               && this->grid_prop[idx_ijk].extension_status == ExtensionStatus::NATURAL) {
            // central difference
            dtype phi_x = (this->phi[this->Index(i + 1, j, k)] - this->phi[this->Index(i - 1, j, k)]) / 2;
            dtype phi_y = (this->phi[this->Index(i, j + 1, k)] - this->phi[this->Index(i, j - 1, k)]) / 2;
            dtype phi_z = (this->phi[this->Index(i, j, k + 1)] - this->phi[this->Index(i, j, k - 1)]) / 2;
            if (phi_x !=0 || phi_y != 0 || phi_z != 0) {
                dtype phi_xx = this->phi[this->Index(i + 1, j, k)] + this->phi[this->Index(i - 1, j, k)] - 2 * this->phi[idx_ijk];
                dtype phi_yy = this->phi[this->Index(i, j + 1, k)] + this->phi[this->Index(i, j - 1, k)] - 2 * this->phi[idx_ijk];
                dtype phi_zz = this->phi[this->Index(i, j, k + 1)] + this->phi[this->Index(i, j, k - 1)] - 2 * this->phi[idx_ijk];
                dtype phi_xy = (this->phi[this->Index(i + 1, j + 1, k)] - this->phi[this->Index(i - 1, j + 1, k)]
                                 - this->phi[this->Index(i + 1, j - 1, k)] + this->phi[this->Index(i - 1, j - 1, k)]) / 4;
                dtype phi_yz = (this->phi[this->Index(i, j + 1, k + 1)] - this->phi[this->Index(i, j - 1, k + 1)]
                                - this->phi[this->Index(i, j + 1, k - 1)] + this->phi[this->Index(i, j - 1, k - 1)]) / 4;
                dtype phi_xz = (this->phi[this->Index(i + 1, j, k + 1)] - this->phi[this->Index(i - 1, j, k + 1)]
                                - this->phi[this->Index(i + 1, j, k - 1)] + this->phi[this->Index(i - 1, j, k - 1)]) / 4;
                val = ((pow(phi_y, 2) + pow(phi_z, 2)) * phi_xx
                                            + (pow(phi_x, 2) + pow(phi_z, 2)) * phi_yy
                                            + (pow(phi_x, 2) + pow(phi_y, 2)) * phi_zz
                                            - 2 * phi_x * phi_y * phi_xy
                                            - 2 * phi_x * phi_z * phi_xz
                                            - 2 * phi_y * phi_z * phi_yz)
                                           / (pow(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z, 1.5));
                this->velocity[idx_ijk] = val;
            }
            else {
                this->velocity[idx_ijk] = 0;
            }
        }
        // extension speed
        else if (this->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY
                    && this->grid_prop[idx_ijk].extension_status == ExtensionStatus::EXTENSION) {
            // ensure upwind scheme
            if (this->phi[idx_ijk] > 0) {
                IdxType argmin_i = this->phi[this->Index(i - 1, j, k)] <= this->phi[this->Index(i + 1, j, k)] ? i - 1 : i + 1;
                IdxType argmin_j = this->phi[this->Index(i, j - 1, k)] <= this->phi[this->Index(i, j + 1, k)] ? j - 1 : j + 1;
                IdxType argmin_k = this->phi[this->Index(i, j, k - 1)] <= this->phi[this->Index(i, j, k + 1)] ? k - 1 : k + 1;
                val = (this->velocity[this->Index(argmin_i, j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(argmin_i, j, k)])
                                           + this->velocity[this->Index(i, argmin_j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, argmin_j, k)])
                                           + this->velocity[this->Index(i, j, argmin_k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, j, argmin_k)]))
                                          / (3 * this->phi[idx_ijk] - this->phi[this->Index(argmin_i, j, k)]
                                             - this->phi[this->Index(i, argmin_j, k)] - this->phi[this->Index(i, j, argmin_k)]);
            }
            else {
                assert(this->phi[idx_ijk] < 0);
                IdxType argmax_i = this->phi[this->Index(i - 1, j, k)] >= this->phi[this->Index(i + 1, j, k)] ? i - 1 : i + 1;
                IdxType argmax_j = this->phi[this->Index(i, j - 1, k)] >= this->phi[this->Index(i, j + 1, k)] ? j - 1 : j + 1;
                IdxType argmax_k = this->phi[this->Index(i, j, k - 1)] >= this->phi[this->Index(i, j, k + 1)] ? k - 1 : k + 1;
                val = (this->velocity[this->Index(argmax_i, j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(argmax_i, j, k)])
                                           + this->velocity[this->Index(i, argmax_j, k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, argmax_j, k)])
                                           + this->velocity[this->Index(i, j, argmax_k)] * (this->phi[idx_ijk] - this->phi[this->Index(i, j, argmax_k)]))
                                          / (3 * this->phi[idx_ijk] - this->phi[this->Index(argmax_i, j, k)]
                                             - this->phi[this->Index(i, argmax_j, k)] - this->phi[this->Index(i, j, argmax_k)]);
            }
            this->velocity[idx_ijk] = val;
        }
    }
}

void Grid3d::Update_velocity()
{
    cout << "Update velocity" << endl;
    FMM3d(this, false);
}

Grid3d* Grid3d::Reinitialize()
{
    cout << "Reinitialization" << endl;
    return FMM3d(this, true);
}

Grid3d* FMM3d(Grid3d *init_grid, bool reinit)
{
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
                    auto idx_ijk = init_grid->Index(i, j, k);
                    // todo: change != boundary to == active
                    if (init_grid->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY) {
                        // in 3-d case, we need to exam 6 directions
                        vector<cv::Vec3i> front_dir(6);
                        if (init_grid->isFrontHere(i, j, k, front_dir) != 0) {
                            //// Maybe, here performance can be improved by check whether abs(phi_val) < a small number.
                            // narrowband status, phival, velocity
                            // Here all stuff including determining value\ sign\ narrowband status can be integrated into one part
                            Determine_front_property(init_grid, new_grid, i, j, k, front_dir);
                            // put >0 and  <0 into different set so that two direction fmm can be done.
                            if (init_grid->phi[idx_ijk] > 0) {
                                close_pq_pos.emplace(PointKeyVal{i, j, k, new_grid->phi[idx_ijk]});
                                new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::OTHER_SIDE;
                            }
                            else if (init_grid->phi[idx_ijk] < 0) {
                                close_pq_neg.emplace(PointKeyVal{i, j, k, new_grid->phi[idx_ijk]});
                                new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::OTHER_SIDE;
                            }
                            else {
                                new_grid->marching_sequence.emplace_back(IndexSet{i, j, k});
//                                new_grid->front.emplace_back(IndexSet{i, j, k});
                            }
//                            new_grid->front.emplace_back(IndexSet{i, j, k});
                            new_grid->grid_prop[idx_ijk].extension_status = ExtensionStatus::NATURAL;
                        }
                    }
                }
            }
        }
    }

    ////// 2. Marching
    new_grid->Marching(close_pq_pos, false);
    new_grid->Marching(close_pq_neg, true);

    ///// post precessing
    new_grid->Build_band();
    new_grid->Extend_velocity();
    if (!reinit) {
        // according to Sethian's paper, we only use new velocity instead of newly initialized phi.
        std::swap(new_grid->velocity, init_grid->velocity);
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
int Determine_front_type(std::vector<cv::Vec3i> &front_dir, std::vector<int> &special_grid_index)
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

void Determine_front_property(Grid3d *old_grid, Grid3d *new_grid, IdxType i, IdxType j, IdxType k,
                              std::vector<cv::Vec3i> &front_dir)
{
    // first deal with phi = 0
    auto idx_ijk = old_grid->Index(i, j, k);
    if (old_grid->phi[idx_ijk] == 0) {
        new_grid->grid_prop[idx_ijk].fmm_status = FMM_Status::ACCEPT;
        new_grid->grid_prop[idx_ijk].nb_status = NarrowBandStatus::ACTIVE;
        new_grid->phi[idx_ijk] = 0;
        return;
    }
    vector<int> special_grid_index(2);
    auto front_type = Determine_front_type(front_dir, special_grid_index);
    dtype dist1, dist2, dist3, dist4, dist5, dist6, temp_dist1, temp_dist2, temp_dist3;
    int s_index1, s_index2, r_index1, r_index2, r_index3, r_index4;
    switch (front_type) {
        // refer to head file for the definition of each type
        case TYPE_A:
            // use point-line distance for approximation
            dist1 = old_grid->phi[idx_ijk] / (old_grid->phi[idx_ijk]
                                              - old_grid->phi[old_grid->Index(i + front_dir[0][0],
                                                                              j + front_dir[0][1],
                                                                              k + front_dir[0][2])]);
            new_grid->phi[idx_ijk] = dist1;
            break;
        case TYPE_B1:
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
        case TYPE_E:
            // precisely compute point-surface distance
            s_index1 = special_grid_index[0];
            r_index1 = (s_index1 + 1) % 5;
            r_index2 = (s_index1 + 2) % 5;
            r_index3 = (s_index1 + 3) % 5;
            r_index4 = (s_index1 + 4) % 5;
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
            break;
        default:
            cerr << "Wrong front type!" << endl;
            exit(1);
    }
}

