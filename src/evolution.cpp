//
// Created by himalaya on 10/29/19.
//

#include "evolution.h"
#include "grid3d.h"
#include <vector>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace std;

int Evolve(BoundingBox &box, vector<Camera> &all_cams)
{
    Grid3d *&grid = box.grid3d;
    PhiCalculator *&velocity_calculator = box.velocity_calculator;
    int counter = 0;
    const int MAX_ITERATION = 1;
//    dtype timestep = 0.1;
    vector<dtype> new_phi(grid->_height * grid->_width * grid->_depth, INF);
    while (counter < MAX_ITERATION) {
        cout << "iteration: " << counter++ << endl;
        auto time_start = omp_get_wtime();
        //// show surface todo: design callback function to always display
//        Show_3D(all_cams, box);
//        fstream fout("testdata.txt", ios::out);
//        for (int z = 0; z < grid->_depth; z++) {
//            fout << "z = " << z << endl;
//            for (int y = 0; y < grid->_width; y++)
//            {fout << scientific << setprecision(3) << setw(5) << setfill('0') << (float)y << " "; }
//            fout << endl;
//            for (int x = 0; x < grid->_height; x++) {
//                for (int y = 0 ; y < grid->_width; y++) {
//                    fout << scientific << setprecision(3) << setw(5) << setfill('0') << grid->phi[grid->Index(x, y, z)] << " ";
//                }
//                fout << endl;
//            }
//            fout << endl;
//        }
//        fout.close();
        bool reinit_flag = false;
#pragma omp parallel for default(none) shared(new_phi, grid, reinit_flag)
        for (IdxType i = grid->band_begin_i; i < grid->band_end_i; i++) {
            dtype timestep = 0.1;
            dtype float_err = 1e-12;
            for (IdxType j = grid->band_begin_j[i]; j < grid->band_end_j[i]; j++) {
                // iterate element in a small piece narrow band
                for (const auto &iter : grid->narrow_band[i][j]) {
                    for (IdxType k = iter.start; k < iter.end; k++) {
                        dtype temp_phi_val;
                        auto idx_ijk = grid->Index(i, j, k);
                        // if re-initialization is required, anything remained should not be changed.
                        if (!reinit_flag) {
                            if (grid->grid_prop[idx_ijk].nb_status != NarrowBandStatus::BOUNDARY /*&& grid->Phi[idx_ijk] != 0*/) {
                                //// todo:capsule central diff part and upwind diff partm
                                // central difference of norm gradient for curvature driven part
                                dtype phi_x = (grid->phi[grid->Index(i + 1, j, k)] - grid->phi[grid->Index(i - 1, j, k)]) / 2;
                                dtype phi_y = (grid->phi[grid->Index(i, j + 1, k)] - grid->phi[grid->Index(i, j - 1, k)]) / 2;
                                //// issue1: I need to increase the boundary distance since here it happened that (i, j, k - 1) is OUTSIDE
                                //// fixed
                                dtype phi_z = (grid->phi[grid->Index(i, j, k + 1)] - grid->phi[grid->Index(i, j, k - 1)]) / 2;
//                                if (isnan(phi_x) || isnan(phi_y) || isnan(phi_z)) {
//                                    cerr << "nan_here" << endl;
//                                    exit(1);
//                                }
                                dtype val = 0;
                                if (phi_x !=0 || phi_y != 0 || phi_z != 0) {
                                    dtype phi_xx = grid->phi[grid->Index(i + 1, j, k)] + grid->phi[grid->Index(i - 1, j, k)] - 2 * grid->phi[idx_ijk];
                                    dtype phi_yy = grid->phi[grid->Index(i, j + 1, k)] + grid->phi[grid->Index(i, j - 1, k)] - 2 * grid->phi[idx_ijk];
                                    dtype phi_zz = grid->phi[grid->Index(i, j, k + 1)] + grid->phi[grid->Index(i, j, k - 1)] - 2 * grid->phi[idx_ijk];
//                                    assert(grid->isValidRange(i + 1, j + 1, k) && grid->isValidRange(i + 1, j - 1, k) && grid->isValidRange(i - 1, j + 1, k) && grid->isValidRange(i - 1, j - 1, k));
                                    dtype phi_xy = (grid->phi[grid->Index(i + 1, j + 1, k)] - grid->phi[grid->Index(i - 1, j + 1, k)]
                                                    - grid->phi[grid->Index(i + 1, j - 1, k)] + grid->phi[grid->Index(i - 1, j - 1, k)]) / 4;
//                                    assert(grid->isValidRange(i, j + 1, k + 1) && grid->isValidRange(i, j + 1, k - 1) && grid->isValidRange(i, j - 1, k + 1) && grid->isValidRange(i, j - 1, k - 1));
                                    dtype phi_yz = (grid->phi[grid->Index(i, j + 1, k + 1)] - grid->phi[grid->Index(i, j - 1, k + 1)]
                                                    - grid->phi[grid->Index(i, j + 1, k - 1)] + grid->phi[grid->Index(i, j - 1, k - 1)]) / 4;
//                                    assert(grid->isValidRange(i + 1, j, k + 1) && grid->isValidRange(i + 1, j, k - 1) && grid->isValidRange(i - 1, j, k + 1) && grid->isValidRange(i - 1, j, k - 1));
                                    dtype phi_xz = (grid->phi[grid->Index(i + 1, j, k + 1)] - grid->phi[grid->Index(i - 1, j, k + 1)]
                                                    - grid->phi[grid->Index(i + 1, j, k - 1)] + grid->phi[grid->Index(i - 1, j, k - 1)]) / 4;
                                    val = ((pow(phi_y, 2) + pow(phi_z, 2)) * phi_xx
                                           + (pow(phi_x, 2) + pow(phi_z, 2)) * phi_yy
                                           + (pow(phi_x, 2) + pow(phi_y, 2)) * phi_zz
                                           - 2 * phi_x * phi_y * phi_xy
                                           - 2 * phi_x * phi_z * phi_xz
                                           - 2 * phi_y * phi_z * phi_yz)
                                          / (pow(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z, 1));
//                                    grid->Phi[idx_ijk] = val;
                                }
                                else {
//                                    grid->Phi[idx_ijk] = 0;
                                    val = 0;
                                }
                                temp_phi_val = grid->phi[idx_ijk] + timestep * val;
                                if (abs(temp_phi_val) <= float_err) {
                                    temp_phi_val = 0;
                                }
                                // if landmine is not hit and the sign of current grid point did not change
                                if (grid->grid_prop[idx_ijk].nb_status != NarrowBandStatus::LANDMINE || temp_phi_val * grid->phi[idx_ijk] >= 0) {
                                    new_phi[idx_ijk] = temp_phi_val;
                                }
                                else {
                                    reinit_flag = true;
                                    new_phi[idx_ijk] = grid->phi[idx_ijk];
                                }
                            }
                            else {
                                // two cases here: 1. velocity = 0, so the value wont change, directly copy
                                //                 2. narrow_bandstatus = BOUNDARY
                                // directly copy lanmine value since it will not ba calculated but only for finding front
                                new_phi[idx_ijk] = grid->phi[idx_ijk];
                            }
                        }
                        else {
                            new_phi[idx_ijk] = grid->phi[idx_ijk];
                        }
                    }
                }
            }
        }
#pragma omp barrier
        std::swap(grid->phi, new_phi);
//        fout.open("testdata.txt", ios::out);
//        for (int z = 0; z < grid->_depth; z++) {
//            fout << "z = " << z << endl;
//            for (int y = 0; y < grid->_width; y++)
//            {fout << scientific << setprecision(3) << setw(5) << setfill('0') << (float)y << " "; }
//            fout << endl;
//            for (int x = 0; x < grid->_height; x++) {
//                for (int y = 0 ; y < grid->_width; y++) {
//                    fout << scientific << setprecision(3) << setw(5) << setfill('0') << grid->phi[grid->Index(x, y, z)] << " ";
//                }
//                fout << endl;
//            }
//            fout << endl;
//        }
//        fout.close();
        // update visibility information
#pragma omp parallel for default(none) shared(box)
        for (int i = 0; i < box.visibility_arr.size(); i++) {
            box.visibility_arr[i].Set_phi(box.grid3d->phi);
            box.visibility_arr[i].Calculate_all();
        }
#pragma omp barrier
        velocity_calculator->Set_phi(grid->phi);
        velocity_calculator->Set_psi(box.visibility_arr);
        if (!reinit_flag) {
            grid->Update_velocity(velocity_calculator);
        }
        else {
            Grid3d * new_grid = grid->Reinitialize(velocity_calculator);
            delete grid;
            grid = new_grid;
        }
        cout << "iteration time cost" << omp_get_wtime() - time_start << endl;
    }
    return MAX_ITERATION;
}