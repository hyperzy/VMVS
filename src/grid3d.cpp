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
        active_distance(4.1), landmine_distance(5.1), boundary_distance(6.1)
{
    unsigned long total_num = height * width * depth;
    this->grid_prop.resize(total_num);
    this->phi.resize(total_num, INF);
    this->velocity.resize(total_num);
    this->narrow_band.resize(height);
    for (auto &iter : this->narrow_band) {iter.resize(width);}
    this->band_begin_i = height;
    this->band_end_i = 0;
    this->band_begin_j = width;
    this->band_end_j = 0;
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
        for (auto j = this->band_begin_j; j < this->band_end_j; j++) {
            bool flag_interior = false;
            IdxType  start = 0, end = 0;
            // the <= is crucial since the flag_interior is changed only when iterated outside the band.
            for (auto k = this->narrow_band[i][j].front().start; k <= this->narrow_band[i][j].front().end; j++) {
                if (!flag_interior) { start = k; end = k;}
                else { end = k;}
                flag_interior = this->grid_prop[this->Index(i, j, k)].nb_status != NarrowBandStatus::OUTSIDE;
                if (start != end && !flag_interior) {
                    this->narrow_band[i][j].emplace_back(NarrowBandExtent{start, end});
                }
            }
            // we need to pop the element recording the coarse narrow band
            this->narrow_band[i][j].pop_front();
        }
    }
    assert(this->band_begin_j <= this->band_end_j && this->band_begin_i <= this->band_end_i);
}

void Grid3d::Build_coarse_band(IdxType i, IdxType j, IdxType k)
{
    // end is one more step of the last element
    this->band_begin_i = i < this->band_begin_i ? i : this->band_begin_i;
    this->band_end_i = i > this->band_end_i - 1 ? i + 1 : this->band_end_i;
    this->band_begin_j = j < this->band_begin_j ? j : this->band_begin_j;
    this->band_end_j = j > this->band_end_j - 1 ? j + 1 : this->band_end_j;
    if (!this->narrow_band[i][j].empty()) {
        auto &start = this->narrow_band[i][j].front().start;
        auto &end = this->narrow_band[i][j].front().end;
        start = k < start ? j : start;
        end = k > end - 1 ? k + 1 : end;
    }
    else {
        this->narrow_band[i][j].emplace_back(NarrowBandExtent{k, (IdxType)(k + 1)});
    }
}

unsigned short Grid3d::isFrontHere(IdxType i, IdxType j, IdxType k) const
{
    unsigned short res = 0;
    auto current_val = this->phi[this->Index(i, j, k)];
    res = (current_val * this->phi[this->Index(i - 1, j, k)] <= 0) |
            (current_val * this->phi[this->Index(i, j - 1, k)] <= 0) << 1u |
            (current_val * this->phi[this->Index(i + 1, j, k)] <= 0) << 2u |
            (current_val * this->phi[this->Index(i, j, k - 1)] <= 0) << 3u |
            (current_val * this->phi[this->Index(i, j + 1, k)] <= 0) << 4u |
            (current_val * this->phi[this->Index(i, j, k + 1)] <= 0) << 5u;
}

void Grid3d::Marching(std::priority_queue<PointKeyVal> &close_set, bool inside)
{
    dtype speed = 1.0;
    dtype reciprocal = 1.0 / speed;
    int index[6][3] = {{-1, 0, 0}, {0, -1, 0}, {1, 0, 0},
                       {0, 0, -1}, {0, 1, 0}, {0, 0, 1}};
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

            // check necessity of the grid to be computed
            if (this->isValidRange(i, j, k) && this->grid_prop[this->Index(i, j, k)].fmm_status != FMM_Status::ACCEPT
                                            && this->grid_prop[this->Index(i, j, k)].fmm_status != FMM_Status::OTHER_SIDE) {
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
                if (pow(c - a, 2) + pow(c - b, 2) >= pow(reciprocal, 2)
                    && abs(b - c) >= reciprocal) {
                    temp = a + reciprocal;
                }
                else if (pow(c - a, 2) + pow(c - b, 2) >= pow(reciprocal, 2)
                         && abs(b - c) < reciprocal) {
                    temp = (b + c + sqrt(2 * pow(reciprocal, 2) - pow(b - c, 2))) / 2;

                }
                else {
                    temp = (a + b + c + sqrt(3 * pow(reciprocal, 2) - pow(c - b, 2) - pow(c - a, 2)
                            - pow(b - a, 2))) / 3;
                }
                // store the minimum value
                this->phi[this->Index(i, j, k)] = min(this->phi[this->Index(i, j, k)], temp);

                // FMM guarantee that the newly inserted value is larger than heap top element.
                // this condition is in case of duplication
                if (this->grid_prop[this->Index(i, j, k)].fmm_status != FMM_Status::CLOSE) {
                    close_set.emplace(PointKeyVal{i, j, k, this->phi[this->Index(i, j, k)]});
                    this->grid_prop[this->Index(i, j, k)].fmm_status = FMM_Status::CLOSE;
                }
                this->grid_prop[this->Index(i, j, k)].extension_status = ExtensionStatus::EXTENSION;
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
    for (IdxType i = 0; i < height; i++) {
        for (IdxType j = 0; j < width; j++) {
            for (auto const &iter : init_grid->narrow_band[i][j]) {
                for (IdxType k = iter.start; k < iter.end; k++) {
                    if (init_grid->grid_prop[init_grid->Index(i, j, k)].nb_status != NarrowBandStatus::BOUNDARY) {
                        unsigned short sign_change = init_grid->isFrontHere(i, j, k);
                        if (sign_change != 0) {
                            //// Maybe, here performance can be improved by check whether abs(phi_val) < a small number.
                            // narrowband status, phival, velocity
                            // Here all stuff including determining value\ sign\ narrowband status can be integrated into one part
                            Determine_front_property(init_grid, new_grid, i, j, k, sign_change);
                            // put >0 and  <0 into different set so that two direction fmm can be done.
                            if (init_grid->phi[init_grid->Index(i, j, k)] > 0) {
                                close_pq_pos.emplace(PointKeyVal{i, j, k, new_grid->phi[new_grid->Index(i, j, k)]});
                                new_grid->grid_prop[new_grid->Index(i, j, k)].fmm_status = FMM_Status::OTHER_SIDE;
                            }
                            else if (init_grid->phi[init_grid->Index(i, j, k)] < 0) {
                                close_pq_neg.emplace(PointKeyVal{i, j, k, new_grid->phi[new_grid->Index(i, j, k)]});
                                new_grid->grid_prop[new_grid->Index(i, j, k)].fmm_status = FMM_Status::OTHER_SIDE;
                            }
                            else {
                                new_grid->marching_sequence.emplace_back(IndexSet{i, j, k});
//                                new_grid->front.emplace_back(IndexSet{i, j, k});
                            }
//                            new_grid->front.emplace_back(IndexSet{i, j, k});
                            new_grid->grid_prop[new_grid->Index(i, j, k)].extension_status = ExtensionStatus::NATURAL;
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

// Return value depending on the construction of 'sign_change'. Refer to function isFrontHere
// For this program:
// LSM {-1, 0, 0} {0, -1, 0} {1, 0, 0} { 0, 0, -1} {0, 1, 0} {0, 0, 1}
void Determine_front_type(unsigned short sign_changed)
{
    // first determine how many 1s 'sign_changed' has
    // and it is definite that 'sign_changed' cannot be 0 or negative due to the caller location
    unsigned short count = 0;
    unsigned short param_copy = sign_changed;
    // to be determined
    bool var1, var2;
    while (sign_changed) {}
}
void Determine_front_property(Grid3d *old_grid, Grid3d *new_grid, IdxType i, IdxType j, IdxType k,
                              unsigned short sign_changed)
{

}

