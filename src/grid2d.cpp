//
// Created by himalaya on 10/6/19.
//

#include "grid2d.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <queue>

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

bool PointKeyVal::operator<(const struct PointKeyVal &rhs) const
{
    return this->phi_val > rhs.phi_val;
}

PointProperty& PointProperty::operator=(const PointProperty &rhs)
{
    this->nb_status = rhs.nb_status;
//    this->phi_val = rhs.phi_val;
    this->coord = rhs.coord;
    return *this;
}
Grid2d::Grid2d(int length)
{
    grid.resize(length);
    for (auto &iter : grid) {
        iter.resize(length);
    }
    narrow_band.resize(length);
    this->band_begin_i = this->grid.size();
    this->band_end_i = 0;
}


void Grid2d::FMM()
{
    ////  Part a. Initialize the shape. This part should be capsuled in another function.
    dtype radius = 5;
    dtype active_bandwidth = 2.1;
    dtype landmine_distance = 3.1;
    int center_i = grid.size() / 2;
    int center_j = grid[0].size() / 2;
#pragma omp parallel for default(none) shared(center_i, center_j, radius, active_bandwidth, landmine_distance)
    for (int i = 0; i < grid.size(); i++) {
        bool flag_interior = false;
        unsigned short start = 0, end = 0;
        for (int j = 0; j < grid[i].size(); j++) {
            grid[i][j].phi_val = sqrt(pow(i - center_i, 2) + pow(j - center_j, 2)) - radius;
            if (!flag_interior) { start = j; end = j;}
            else { end = j;}

            if (abs(grid[i][j].phi_val) <= landmine_distance) {
                flag_interior = true;
                if (abs(grid[i][j].phi_val) <= active_bandwidth) {
                    grid[i][j].nb_status = NarrowBandStatus::ACTIVE;
                }
                else {
                    grid[i][j].nb_status = NarrowBandStatus::LANDMINE;
                }
            }
            else { flag_interior = false;}
            // an interval recorded

            if (start != end && !flag_interior) {
//                #pragma omp critical
                {
                    narrow_band[i].emplace_back(NarrowBandBound{start, end});
                }
            }
        }
    }
#pragma omp barrier
}

bool Grid2d::isValidRange(unsigned short i, unsigned short j)
{
    return (i >= 0 && i < this->height && j >= 0 && j < this->width);
}

void Grid2d::Marching(std::priority_queue<PointKeyVal> &close_set, dtype active_bandwidth, dtype landmine_distance, bool inside)
{
    dtype speed = 1.0;
    dtype reciprocal = 1.0 / speed;
    this->height = this->grid.size();
    this->width = this->grid[0].size();
    int index[4][2] = {-1, 0, 0, -1, 1, 0, 0, 1};
    vector<IndexPair> band_point_index;
    while (!close_set.empty()) {
        PointKeyVal point = close_set.top();
        if (point.phi_val <= active_bandwidth) { this->grid[point.i][point.j].nb_status = NarrowBandStatus::ACTIVE; }
        else if (point.phi_val <= landmine_distance) {this->grid[point.i][point.j].nb_status = NarrowBandStatus::LANDMINE; }
        else {
            // it means all the points remained are outside narrow band. So the loop can break.
            this->grid[point.i][point.j].nb_status = NarrowBandStatus::OUTSIDE;
            break;
        }
        // add new CLOSE point from FAR point and computing the value
        for (const auto &idx_iter : index) {
            unsigned short index_i = point.i + idx_iter[0];
            unsigned short index_j = point.j + idx_iter[1];
            if (this->isValidRange(index_i, index_j) && this->grid[index_i][index_j].fmm_status != FMM_Status::ACCEPT
                                                        && this->grid[index_i][index_j].fmm_status != FMM_Status::OTHER_SIDE) {
                // smaller value in x direction
                dtype smaller_val_x = min(this->isValidRange(index_i, index_j - 1) ? this->grid[index_i][index_j - 1].phi_val : INF,
                                          this->isValidRange(index_i, index_j + 1) ? this->grid[index_i][index_j + 1].phi_val : INF);
                dtype smaller_val_y = min(this->isValidRange(index_i - 1, index_j) ? this->grid[index_i - 1][index_j].phi_val : INF,
                                          this->isValidRange(index_i + 1, index_j) ? this->grid[index_i + 1][index_j].phi_val : INF);
                if (abs(smaller_val_x - smaller_val_y) >= reciprocal) {
                    this->grid[index_i][index_j].phi_val = min(min(smaller_val_x, smaller_val_y) + reciprocal,
                                                                  this->grid[index_i][index_j].phi_val);
                }
                else {
                    this->grid[index_i][index_j].phi_val = std::min(dtype((smaller_val_x + smaller_val_y +
                                                                        sqrt(2 * pow(reciprocal, 2) - pow(smaller_val_x - smaller_val_y, 2))) / 2),
                                                                       this->grid[index_i][index_j].phi_val);
                }
                // FMM guarantee that the newly inserted value is larger than heap top element.
                // this condition is in case of duplication
                if (this->grid[index_i][index_j].fmm_status != FMM_Status::CLOSE) {
                    close_set.emplace(PointKeyVal{index_i, index_j, this->grid[index_i][index_j].phi_val});
                    this->grid[index_i][index_j].fmm_status = FMM_Status::CLOSE;
                }
                this->grid[index_i][index_j].extension_status = ExtensionStatus::EXTENSION;
            }
        }
//        this->Extend_velocity(this->grid[point.i][point.j].fmm_status, point.i, point.j);
        this->grid[point.i][point.j].fmm_status = FMM_Status::ACCEPT;
        this->marching_sequence.emplace_back(IndexPair{point.i, point.j});
        // add current point into (pos/neg) narrow band
        band_point_index.emplace_back(IndexPair{point.i, point.j});
        this->Build_coarse_band(point.i, point.j);
        close_set.pop();
    }
    if (inside)
        for (const auto &iter : band_point_index) {
            this->grid[iter.i][iter.j].phi_val *= -1;
        }
}

void Zero_val_handler(Grid2d &old_grid, Grid2d &new_grid,
                        unsigned short i, unsigned short j,
                        std::priority_queue<PointKeyVal> &close_pq_pos,
                        std::priority_queue<PointKeyVal> &close_pq_neg)
{
    new_grid.grid[i][j].phi_val = 0;
    new_grid.grid[i][j].fmm_status = FMM_Status::ACCEPT;
    new_grid.grid[i][j].nb_status = NarrowBandStatus::ACTIVE;
    new_grid.marching_sequence.emplace_back(IndexPair{i, j});
    new_grid.grid[i][j].extension_status = ExtensionStatus::NATURAL;
    if (i != 0 && old_grid.isFrontHere(i - 1, j) == 0 && old_grid.grid[i - 1][j].phi_val != 0) {
        old_grid.grid[i - 1][j].phi_val < 0 ? close_pq_neg.emplace(PointKeyVal{(unsigned short)(i - 1), j, 1}) :
                                    close_pq_pos.emplace(PointKeyVal{(unsigned short)(i - 1), j, 1});
        new_grid.grid[i - 1][j].fmm_status = FMM_Status::CLOSE;
        new_grid.grid[i - 1][j].phi_val = 1;

    }
    if (i != new_grid.grid.size() && old_grid.isFrontHere(i + 1, j) == 0 && old_grid.grid[i + 1][j].phi_val != 0) {
        old_grid.grid[i + 1][j].phi_val < 0 ? close_pq_neg.emplace(PointKeyVal{(unsigned short)(i + 1), j, 1}) :
                                    close_pq_pos.emplace(PointKeyVal{(unsigned short)(i + 1), j, 1});
        new_grid.grid[i + 1][j].fmm_status = FMM_Status::CLOSE;
        new_grid.grid[i + 1][j].phi_val = 1;
    }
    if (j != 0 && old_grid.isFrontHere(i, j - 1) == 0 && old_grid.grid[i][j - 1].phi_val != 0) {
        old_grid.grid[i][j - 1].phi_val < 0 ? close_pq_neg.emplace(PointKeyVal{i, (unsigned short)(j - 1), 1}) :
                                        close_pq_pos.emplace(PointKeyVal{i, (unsigned short)(j - 1), 1});
        new_grid.grid[i][j - 1].fmm_status = FMM_Status::CLOSE;
        new_grid.grid[i][j - 1].phi_val = 1;
    }
    if (j != new_grid.grid[i].size() && old_grid.isFrontHere(i, j + 1) == 0 && old_grid.grid[i][j + 1].phi_val != 0) {
        old_grid.grid[i][j + 1].phi_val < 0 ? close_pq_neg.emplace(PointKeyVal{i, (unsigned short)(j + 1), 1}) :
                                        close_pq_pos.emplace(PointKeyVal{i, (unsigned short)(j + 1), 1});
        new_grid.grid[i][j + 1].fmm_status = FMM_Status::CLOSE;
        new_grid.grid[i][j + 1].phi_val = 1;
    }
//    new_grid.Extend_velocity(new_grid.grid[i][j].fmm_status, i, j);
}

void Grid2d::Build_coarse_band(unsigned short i, unsigned short j)
{
    this->band_begin_i = i < this->band_begin_i ? i : this->band_begin_i;
    this->band_end_i = i >= this->band_end_i - 1 ? i + 1 : this->band_end_i;
    if (!this->narrow_band[i].empty()) {
        auto &start = this->narrow_band[i].front().start;
        auto &end = this->narrow_band[i].front().end;
        start = j < start ? j : start;
        end = j >= end - 1 ? j + 1 : end;
    }
    else {
        this->narrow_band[i].emplace_back(NarrowBandBound{j, (unsigned short)(j + 1)});
    }
}

void Grid2d::Build_band()
{
//    #pragma omp parallel for default(none)
    for (auto i = this->band_begin_i; i < this->band_end_i; i++) {
        bool flag_interior = false;
        unsigned short start = 0, end = 0;
        // the <= is crucial since the flag_interior is changed only when iterated outside the band.
        for (auto j = this->narrow_band[i].front().start; j <= this->narrow_band[i].front().end; j++) {
            if (!flag_interior) { start = j; end = j;}
            else {end = j;}
            flag_interior = this->grid[i][j].nb_status != NarrowBandStatus::OUTSIDE;
            if (start != end && !flag_interior) {
                this->narrow_band[i].emplace_back(NarrowBandBound{start, end});
            }
        }
        this->narrow_band[i].pop_front();
    }
}

void Grid2d::Approx_front()
{
    cv::Mat out(grid.size(), grid.size(), CV_8UC1, Scalar(0));
    bool **mask = new bool*[grid.size()] ;
    for (int i = 0; i < grid.size(); i++) {
        mask[i] = new bool[grid.size()];
        memset(mask[i], false, grid.size() * sizeof(false));
    }

//    for (unsigned long i = 0; i < grid.size(); i++) {
//        for (unsigned long j = 0; j < grid.size(); j++) {
//            cout << mask[i][j] << " ";
//        }
//        cout << endl;
//    }


    int index[4][2] = {-1, 0, 0, -1, 1, 0, 0, 1};
    for (unsigned long i = 0; i < grid.size(); i++) {
        for (auto &iter : narrow_band[i]) {
            for (unsigned long j = iter.start; j < iter.end; j++) {
                if (grid[i][j].nb_status == NarrowBandStatus::LANDMINE) continue;
                else if (abs(grid[i][j].phi_val) < 1e-4) mask[i][j] = true;
                else {
                    unsigned short sign_change = isFrontHere(i, j);
                    int index_iter = 0;
                    while (sign_change != 0) {
                        if (sign_change & 0x01u) {
                            // smaller value means the front is closer to this grid center.
                            if (abs(grid[i][j].phi_val) > abs(grid[i + index[index_iter][0]][j + index[index_iter][1]].phi_val))
                                mask[i+ index[index_iter][0]][j + index[index_iter][1]] = true;
                            else
                                mask[i][j] = true;
                        }
                        sign_change = sign_change >> 1u;
                        index_iter++;
                    }
                }
            }
        }
    }

//    for (unsigned long i = 0; i < grid.size(); i++) {
//        for (unsigned long j = 0; j < grid.size(); j++) {
//            cout << mask[i][j] << " ";
//        }
//        cout << endl;
//    }

    for (auto i = this->band_begin_i; i < this->band_end_i; i++) {
        for (const auto &iter : this->narrow_band[i]) {
            for (auto j = iter.start; j < iter.end; j++) {
                switch(this->grid[i][j].nb_status) {
                    case NarrowBandStatus::LANDMINE:
                        out.at<uchar>(i, j) = 80;
                        break;
                    case NarrowBandStatus::ACTIVE:
                        out.at<uchar>(i, j) = 160;
                        break;
                }
            }
        }
    }

    Mat for_show(grid.size(), grid.size(), CV_8UC1, Scalar(255));
    Mat mask_mat(grid.size(), grid.size(), CV_8UC1);
    for (unsigned short i = 0; i < grid.size(); i++) {
        Mat dst_temp = mask_mat.row(i);
        Mat src_temp(1, grid.size(), CV_8UC1, mask[i]);
        src_temp.copyTo(dst_temp);
    }
    for_show.copyTo(out, mask_mat);
//    cout << mask_mat << endl;
    namedWindow("show front", 0);
    imshow("show front", out);
    waitKey(1);

    for (unsigned long i = 0; i < grid.size(); i++) {
        delete [] mask[i];
    }
    delete [] mask;
}

unsigned short Grid2d::isFrontHere(unsigned long i, unsigned long j)
{
    unsigned short res = 0;
    res =  (grid[i][j].phi_val * grid[i - 1][j].phi_val <= 0) |
            (grid[i][j].phi_val * grid[i][j - 1].phi_val <= 0) << 1u |
            (grid[i][j].phi_val * grid[i + 1][j].phi_val <= 0) << 2u |
            (grid[i][j].phi_val * grid[i][j + 1].phi_val <= 0) << 3u;
    return res;
}

// Return value depends on the construction rule of sign_changed
unsigned short Determine_front_type(unsigned short sign_changed)
{
    // first determine how many 1s is there of sign_changed
    // and it's sure that sign_changed cannot be 0 or negative.
    unsigned short count = 0;
    unsigned short param_copy = sign_changed;
    bool consecutive_pre = false, consecutive_done = false;
    while (sign_changed) {
        if (sign_changed & 1u) {
            count++;
            if (consecutive_pre) consecutive_done = true;
            consecutive_pre = true;
        }
        else { consecutive_pre = false;}
        sign_changed = sign_changed >> 1u;
    }

    switch (count) {
        case 1: return FrontType::TYPE_A;
        case 2:
            if (consecutive_done || param_copy == 0x09u) return TYPE_B;
            else return TYPE_D;
        case 3: return TYPE_C;
        case 4: return TYPE_E;
        default:
            cerr << "wrong bit 1 number. Check your code" << endl;
            exit(1);
    }

}

// the lowest position when first n consecutive 1s occurs. (circular)
int Pos_1_occur(unsigned sign_changed, int n)
{
    int pos = 0;
    // 4 for 2d, 6 for 3d
    sign_changed = sign_changed | sign_changed << 4u;
    while (sign_changed) {
        switch (n) {
            case 1:
                if ((sign_changed & 0x01u) == 0x01u) return pos;
                break;
            case 2:
                if ((sign_changed & 0x03u) == 0x03u) return pos;
                break;
            case 3:
                if ((sign_changed & 0x07u) == 0x07u) return pos;
                break;
            case 4:
                if ((sign_changed & 0xffu) == 0xffu) return pos;
            default:
                cerr << "error signed change" << endl;
                exit(1);
        }
        pos++;
        sign_changed = sign_changed >> 1u;
    }
}
void Determine_front_property(Grid2d *old_grid, Grid2d *new_grid,
                                unsigned short i, unsigned short j,
                                unsigned short sign_changed)
{
    if (old_grid->grid[i][j].phi_val == 0) {
        new_grid->grid[i][j].fmm_status = FMM_Status::ACCEPT;
        new_grid->grid[i][j].nb_status = NarrowBandStatus::ACTIVE;
        new_grid->grid[i][j].phi_val = 0;
        return;
    }
    auto front_type = Determine_front_type(sign_changed);
    int index[4][2] = {-1, 0, 0, -1, 1, 0, 0, 1};
    int pos1;
    dtype dist1;
    dtype dist2, dist3, dist4, temp_dist1, temp_dist2;
    int pos2, pos3, pos4;
    switch (front_type) {
        case TYPE_A:
            pos1 = Pos_1_occur(sign_changed, 1);
            dist1 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos1][0]][j + index[pos1][1]].phi_val);
            new_grid->grid[i][j].phi_val = dist1;
            break;
        case TYPE_B:
            pos1 = Pos_1_occur(sign_changed, 2);
            dist1 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos1][0]][j + index[pos1][1]].phi_val);
            // the second related position
            // 4 for 2d, 6 for 3d
            pos2 = (pos1 + 1) % 4;
            dist2 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos2][0]][j + index[pos2][1]].phi_val);
            new_grid->grid[i][j].phi_val = sqrt(pow(dist1 * dist2, 2) / (pow(dist1, 2) + pow(dist2, 2)));
            break;
        case TYPE_C:
            pos1 = Pos_1_occur(sign_changed, 3);
            dist1 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos1][0]][j + index[pos1][1]].phi_val);
            // the second related position
            // 4 for 2d, 6 for 3d
            pos2 = (pos1 + 1) % 4;
            dist2 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos2][0]][j + index[pos2][1]].phi_val);
            // the third related position
            // 4 for 2d, 6 for 3d
            pos3 = (pos1 + 2) % 4;
            dist3 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos3][0]][j + index[pos3][1]].phi_val);
            temp_dist1 = min(dist1, dist3);
            new_grid->grid[i][j].phi_val = sqrt(pow(temp_dist1 * dist2, 2) / (pow(temp_dist1, 2) + pow(dist2, 2)));
            break;
        case TYPE_D:
            pos1 = Pos_1_occur(sign_changed, 2);
            dist1 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos1][0]][j + index[pos1][1]].phi_val);
            pos2 = (pos1 + 2) % 4;
            dist2 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos2][0]][j + index[pos2][1]].phi_val);
            new_grid->grid[i][j].phi_val = min(dist1, dist2);
            break;
        case TYPE_E:
            pos1 = Pos_1_occur(sign_changed, 4);
            dist1 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos1][0]][j + index[pos1][1]].phi_val);
            pos2 = (pos1 + 1) % 4;
            pos3 = (pos1 + 2) % 4;
            pos4 = (pos1 + 3) % 4;
            dist2 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos2][0]][j + index[pos2][1]].phi_val);
            dist3 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos3][0]][j + index[pos3][1]].phi_val);
            dist4 = old_grid->grid[i][j].phi_val / (old_grid->grid[i][j].phi_val - old_grid->grid[i + index[pos4][0]][j + index[pos4][1]].phi_val);
            temp_dist1 = min(dist1, dist3);
            temp_dist2 = min(dist2, dist4);
            new_grid->grid[i][j].phi_val = sqrt(pow(temp_dist1 * temp_dist2, 2) / (pow(temp_dist1, 2) + pow(temp_dist2, 2)));
            break;
        default:
            cerr << "Error front type. Check the Code!" << endl;
            exit(1);
    }
//    new_grid[i][j].fmm_status = FMM_Status::OTHER_SIDE;
}

// this function can be done parallelly
void Grid2d::Extend_velocity(int fmm_status, unsigned short i, unsigned short j)
{
    //// here I did not extend curvature from nodal points to grid points
//    // first extend velocity at front
//    for (const auto &iter : this->front) {
//        auto i = iter.i;
//        auto j = iter.j;
//        dtype phi_x = (this->grid[i][j + 1].phi_val - this->grid[i][j - 1].phi_val) / 2;
//        dtype phi_y = (this->grid[i + 1][j].phi_val - this->grid[i - 1][j].phi_val) / 2;
//        if (phi_x != 0 || phi_y != 0) {
//            dtype phi_xx = this->grid[i][j + 1].phi_val + this->grid[i][j - 1].phi_val - 2 * this->grid[i][j].phi_val;
//            dtype phi_yy = this->grid[i + 1][j].phi_val + this->grid[i - 1][j].phi_val - 2 * this->grid[i][j].phi_val;
//            dtype phi_xy = (this->grid[i + 1][j + 1].phi_val - this->grid[i - 1][j + 1].phi_val
//                            - this->grid[i + 1][j - 1].phi_val + this->grid[i - 1][j - 1].phi_val) / 4;
//            this->grid[i][j].velocity = (pow(phi_x, 2) * phi_yy + pow(phi_y, 2) * phi_xx - 2 * phi_x * phi_y * phi_xy) / pow(phi_x * phi_x + phi_y * phi_y, 1.5);
//        }
//        else {
//            this->grid[i][j].velocity = 0;
//        }
//    }

    // initial condition
    if (fmm_status == FMM_Status::ACCEPT || fmm_status == FMM_Status::OTHER_SIDE) {
        dtype phi_x = (this->grid[i][j + 1].phi_val - this->grid[i][j - 1].phi_val) / 2;
        dtype phi_y = (this->grid[i + 1][j].phi_val - this->grid[i - 1][j].phi_val) / 2;
        if (phi_x != 0 || phi_y != 0) {
            dtype phi_xx = this->grid[i][j + 1].phi_val + this->grid[i][j - 1].phi_val - 2 * this->grid[i][j].phi_val;
            dtype phi_yy = this->grid[i + 1][j].phi_val + this->grid[i - 1][j].phi_val - 2 * this->grid[i][j].phi_val;
            dtype phi_xy = (this->grid[i + 1][j + 1].phi_val - this->grid[i - 1][j + 1].phi_val
                            - this->grid[i + 1][j - 1].phi_val + this->grid[i - 1][j - 1].phi_val) / 4;
            this->grid[i][j].velocity = (pow(phi_x, 2) * phi_yy + pow(phi_y, 2) * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                                        pow(phi_x * phi_x + phi_y * phi_y, 1.5);
        }
        else {
            this->grid[i][j].velocity = 0;
        }
    }
    // extension
    else {
        // upwind scheme
        unsigned short argmin_i = this->grid[i - 1][j].phi_val <= this->grid[i + 1][j].phi_val ? i - 1 : i + 1;
        unsigned short argmin_j = this->grid[i][j - 1].phi_val <= this->grid[i][j + 1].phi_val ? j - 1 : j + 1;
        this->grid[i][j].velocity = (this->grid[argmin_i][j].phi_val * (this->grid[i][j].phi_val - this->grid[argmin_i][j].phi_val) +
                                     this->grid[i][argmin_j].phi_val * (this->grid[i][j].phi_val - this->grid[i][argmin_j].phi_val)) /
                                    (2 * this->grid[i][j].phi_val - this->grid[argmin_i][j].phi_val - this->grid[i][argmin_j].phi_val);
    }
}

void Grid2d::Extend_velocity()
{
//#pragma omp parallel for default(none)
    for (unsigned long idx = 0; idx < this->marching_sequence.size(); idx++) {
        auto &iter = this->marching_sequence[idx];
        if (this->grid[iter.i][iter.j].nb_status == NarrowBandStatus::ACTIVE
        && this->grid[iter.i][iter.j].extension_status == ExtensionStatus::NATURAL) {
            dtype phi_x = (this->grid[iter.i][iter.j + 1].phi_val - this->grid[iter.i][iter.j - 1].phi_val) / 2;
            dtype phi_y = (this->grid[iter.i + 1][iter.j].phi_val - this->grid[iter.i - 1][iter.j].phi_val) / 2;
            if (phi_x != 0 || phi_y != 0) {
                dtype phi_xx = this->grid[iter.i][iter.j + 1].phi_val + this->grid[iter.i][iter.j - 1].phi_val - 2 * this->grid[iter.i][iter.j].phi_val;
                dtype phi_yy = this->grid[iter.i + 1][iter.j].phi_val + this->grid[iter.i - 1][iter.j].phi_val - 2 * this->grid[iter.i][iter.j].phi_val;
                dtype phi_xy = (this->grid[iter.i + 1][iter.j + 1].phi_val - this->grid[iter.i - 1][iter.j + 1].phi_val
                                - this->grid[iter.i + 1][iter.j - 1].phi_val + this->grid[iter.i - 1][iter.j - 1].phi_val) / 4;
                this->grid[iter.i][iter.j].velocity = (pow(phi_x, 2) * phi_yy + pow(phi_y, 2) * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                                            pow(phi_x * phi_x + phi_y * phi_y, 1.5);
            }
            else {
                this->grid[iter.i][iter.j].velocity = 0;
            }
        }
        else if (this->grid[iter.i][iter.j].nb_status == NarrowBandStatus::ACTIVE
                 && this->grid[iter.i][iter.j].extension_status == ExtensionStatus::EXTENSION) {
            if (this->grid[iter.i][iter.j].phi_val > 0) {
                unsigned short argmin_i = this->grid[iter.i - 1][iter.j].phi_val <= this->grid[iter.i + 1][iter.j].phi_val ? iter.i - 1 : iter.i + 1;
                unsigned short argmin_j = this->grid[iter.i][iter.j - 1].phi_val <= this->grid[iter.i][iter.j + 1].phi_val ? iter.j - 1 : iter.j + 1;
                this->grid[iter.i][iter.j].velocity = (this->grid[argmin_i][iter.j].velocity * (this->grid[iter.i][iter.j].phi_val - this->grid[argmin_i][iter.j].phi_val) +
                                                       this->grid[iter.i][argmin_j].velocity * (this->grid[iter.i][iter.j].phi_val - this->grid[iter.i][argmin_j].phi_val)) /
                                                      (2 * this->grid[iter.i][iter.j].phi_val - this->grid[argmin_i][iter.j].phi_val - this->grid[iter.i][argmin_j].phi_val);
            }
            else {
                unsigned short argmax_i = this->grid[iter.i - 1][iter.j].phi_val >= this->grid[iter.i + 1][iter.j].phi_val ? iter.i - 1 : iter.i + 1;
                unsigned short argmax_j = this->grid[iter.i][iter.j - 1].phi_val >= this->grid[iter.i][iter.j + 1].phi_val ? iter.j - 1 : iter.j + 1;
                this->grid[iter.i][iter.j].velocity = (this->grid[argmax_i][iter.j].velocity * (this->grid[iter.i][iter.j].phi_val - this->grid[argmax_i][iter.j].phi_val) +
                                                       this->grid[iter.i][argmax_j].velocity * (this->grid[iter.i][iter.j].phi_val - this->grid[iter.i][argmax_j].phi_val)) /
                                                      (2 * this->grid[iter.i][iter.j].phi_val - this->grid[argmax_i][iter.j].phi_val - this->grid[iter.i][argmax_j].phi_val);
            }
        }
    }
}

void Grid2d::clear()
{
    this->narrow_band.clear();

}

Grid2d* FMM(Grid2d *init_grid)
{
//    init_grid->FMM();

//    cout << endl;
//    for (unsigned long i = 0; i < init_grid->grid.size(); i++) {
//        for(unsigned long j = 0; j < init_grid->grid[i].size(); j++) {
//            cout << scientific << setprecision(3) << setw(5) << setfill('0') << right << init_grid->grid[i][j].phi_val << " ";
//        }
////        if (narrow_band[i].empty()) {
////            cout << "Not in band" << endl;
////            continue;
////        }
////        for (auto &iter : narrow_band[i]) {
////            cout << iter.start << " " << iter.end;
////        }
//        cout << endl;
//    }

    //// FMM
    ///// FMM init: find front
    // parallel programming can be applied here
    Grid2d *new_grid = new Grid2d(init_grid->grid.size());
    // index of points where front lie inside
    vector<IndexPair> pos_val_front_index;
    vector<IndexPair> neg_val_front_index;
    // a heap sort data structure to store grids to be processed.
    priority_queue<PointKeyVal> close_pq_pos;
    priority_queue<PointKeyVal> close_pq_neg;
    for (unsigned short i = 0; i < init_grid->grid.size(); i++) {
        for (auto &iter : init_grid->narrow_band[i]) {
            for (unsigned short j = iter.start; j < iter.end; j++) {
                if (init_grid->grid[i][j].nb_status == NarrowBandStatus::ACTIVE) {
                    unsigned short sign_changed = init_grid->isFrontHere(i, j);
                    if (sign_changed != 0) {
                        //// Maybe, here performance can be improved by check whether abs(phi_val) < a small number.
                        // narrowband status, phival, velocity
                        // Here all stuff including determining value\ sign\ narrowband status can be integrated into one part
                        Determine_front_property(init_grid, new_grid, i, j, sign_changed);
                        new_grid->grid[i][j] = init_grid->grid[i][j];
                        // put >0 and  <0 into different set so that two direction fmm can be done.
                        if (init_grid->grid[i][j].phi_val > 0) {
                            close_pq_pos.emplace(PointKeyVal{i, j, new_grid->grid[i][j].phi_val});
                            new_grid->grid[i][j].fmm_status = FMM_Status::OTHER_SIDE;
                        } else if (init_grid->grid[i][j].phi_val < 0) {
                            close_pq_neg.emplace(PointKeyVal{i, j, new_grid->grid[i][j].phi_val});
                            new_grid->grid[i][j].fmm_status = FMM_Status::OTHER_SIDE;
                        } else {
                            new_grid->marching_sequence.emplace_back(IndexPair{i, j});
                            new_grid->front.emplace_back(IndexPair{i, j});
                        }
                        new_grid->front.emplace_back(IndexPair{i, j});
                        new_grid->grid[i][j].extension_status = ExtensionStatus::NATURAL;
                    }
//                else if (this->grid[i][j].phi_val == 0) {
//                    Zero_val_handler(*this, new_grid, i, j, close_pq_pos, close_pq_neg);
//                    new_grid.front.emplace_back(IndexPair{i, j});
//                }
                }
            }
        }
    }

    new_grid->narrow_band.clear();
    new_grid->narrow_band.resize(new_grid->grid.size());

//    cout << endl;
//    for (unsigned long i = 0; i < new_grid->grid.size(); i++) {
//        for(unsigned long j = 0; j < new_grid->grid[i].size(); j++) {
//            cout << scientific << setprecision(3) << setw(5) << setfill('0') << right << new_grid->grid[i][j].phi_val << " ";
//        }
////        if (narrow_band[i].empty()) {
////            cout << "Not in band" << endl;
////            continue;
////        }
////        for (auto &iter : narrow_band[i]) {
////            cout << iter.start << " " << iter.end;
////        }
//        cout << endl;
//    }

    //// FMM marching
    dtype active_bandwidth = 2.1;
    dtype landmine_distance = 3.1;
    new_grid->Marching(close_pq_pos, active_bandwidth, landmine_distance, false);
    new_grid->Marching(close_pq_neg, active_bandwidth, landmine_distance, true);
    cout << endl;

//    for (unsigned long i = 0; i < new_grid->grid.size(); i++) {
//        for(unsigned long j = 0; j < new_grid->grid[i].size(); j++) {
//            cout << scientific << setprecision(3) << setw(5) << setfill('0') << right << new_grid->grid[i][j].phi_val << " ";
//        }
////        if (new_grid->narrow_band[i].empty()) {
////            cout << "Not in band" << endl;
////            continue;
////        }
////        for (auto &iter : new_grid->narrow_band[i]) {
////            cout << iter.start << " " << iter.end;
////        }
//        cout << endl;
//    }
//    cout << endl;

    new_grid->Build_band();

    new_grid->Extend_velocity();
//    for (unsigned long i = 0; i < new_grid->grid.size(); i++) {
//        for(unsigned long j = 0; j < new_grid->grid[i].size(); j++) {
//            cout << scientific << setprecision(3) << setw(5) << setfill('0') << right << new_grid->grid[i][j].velocity << " ";
//        }
////        if (narrow_band[i].empty()) {
////            cout << "Not in band" << endl;
////            continue;
////        }
////        for (auto &iter : narrow_band[i]) {
////            cout << iter.start << " " << iter.end;
////        }
//        cout << endl;
//    }
    new_grid->Approx_front();
    return new_grid;
}

void Evolve(Grid2d *old_grid)
{
    double start, end;
    int counter = 0;
    dtype timestep = 0.5;
    while (counter < 1500) {
        start = clock();
        cout << "iteration : " << counter << endl;
        Grid2d *new_grid = new Grid2d(old_grid->grid.size());
//#pragma omp parallel for default(none) schedule(dynamic, 5) shared(timestep, new_grid, old_grid)
        for (unsigned i = old_grid->band_begin_i; i < old_grid->band_end_i; i++) {
            for (const auto &iter : old_grid->narrow_band[i]) {
                for (auto j = iter.start; j < iter.end; j++) {
                    if (old_grid->grid[i][j].nb_status == ACTIVE && old_grid->grid[i][j].velocity != 0) {
                        // which means phi_x and phi_y are not zeros
                        dtype phi_x = (old_grid->grid[i][j + 1].phi_val - old_grid->grid[i][j - 1].phi_val) / 2;
                        dtype phi_y = (old_grid->grid[i + 1][j].phi_val - old_grid->grid[i - 1][j].phi_val) / 2;
                        new_grid->grid[i][j].phi_val = old_grid->grid[i][j].phi_val + timestep * old_grid->grid[i][j].velocity * sqrt(pow(phi_x, 2) + pow(phi_y, 2));
                    }
                    else { // two cases here: 1. velocity = 0, so the value wont change, directly copy
                           //                 2. narrow_bandstatus = LANDMINE
                        // directly copy lanmine value since it will not ba calculated but only for finding front
                        new_grid->grid[i][j].phi_val = old_grid->grid[i][j].phi_val;
                    }
                    new_grid->grid[i][j].nb_status = old_grid->grid[i][j].nb_status;

                }
            }
        }
        new_grid->narrow_band = move(old_grid->narrow_band);
        auto newest_grid = FMM(new_grid);
        newest_grid->Approx_front();
        delete new_grid;
        delete old_grid;
        old_grid = newest_grid;
        old_grid->Approx_front();
        counter++;
        end = clock();
        cout << (end - start) / CLOCKS_PER_SEC << endl;
    }


}


