//
// Created by himalaya on 10/29/19.
//

#include "evolution.h"
#include "grid3d.h"
#include "display.h"
#include <vector>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace std;

int Evolve(BoundingBox &box, vector<Camera> &all_cams)
{
    static unsigned long COUNTER = 1;
    Grid3d *&grid = box.grid3d;
    PhiCalculator *&velocity_calculator = box.velocity_calculator;
    int counter = 0;
    const int MAX_ITERATION = 50;
//    dtype timestep = 0.1;
    vector<dtype> new_phi(grid->_height * grid->_width * grid->_depth, INF);
    while (counter < MAX_ITERATION) {
        COUNTER++;
        cout << "iteration: " << counter++ << endl;
        auto time_start = omp_get_wtime();
        //// show surface todo: design callback function to always display
#if USE_NEW
#else
        Show_3D(all_cams, box);
#endif
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
//        fout.open("testdata_v.txt", ios::out);
//        for (int z = 0; z < grid->_depth; z++) {
//            fout << "z = " << z << endl;
//            for (int y = 0; y < grid->_width; y++)
//            {fout << scientific << setprecision(3) << setw(5) << setfill('0') << (float)y << " "; }
//            fout << endl;
//            for (int x = 0; x < grid->_height; x++) {
//                for (int y = 0 ; y < grid->_width; y++) {
//                    fout << scientific << setprecision(3) << setw(5) << setfill('0') << grid->Phi[grid->Index(x, y, z)] << " ";
//                }
//                fout << endl;
//            }
//            fout << endl;
//        }
//        fout.close();
        bool reinit_flag = false;
#pragma omp parallel for default(none) shared(new_phi, grid, reinit_flag)
        for (IdxType i = grid->band_begin_i; i < grid->band_end_i; i++) {
#if USE_SIL
            dtype timestep = 0.1;
#else
            dtype timestep = 0.01;
#endif
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
                                dtype val = Compute_delta(grid, i, j, k);
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
        if (COUNTER % 128 == 0)
            reinit_flag = true;
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
        if (!reinit_flag) {
#if USE_SIL
#else
#pragma omp parallel for default(none) shared(box)
            for (int i = 0; i < box.visibility_arr.size(); i++) {
                box.visibility_arr[i].Set_phi(box.grid3d->phi);
                box.visibility_arr[i].Calculate_all();
            }
#pragma omp barrier
            velocity_calculator->Set_phi(grid->phi);
            velocity_calculator->Set_psi(box.visibility_arr);
#endif
            grid->Update_velocity(velocity_calculator);
        }
        else {
#if USE_SIL
            Grid3d * new_grid = grid->Reinitialize(velocity_calculator);
            delete grid;
            grid = new_grid;
#else
            Grid3d * new_grid = grid->Reinitialize(nullptr);
            delete grid;
            grid = new_grid;
#pragma omp parallel for default(none) shared(box)
            for (int i = 0; i < box.visibility_arr.size(); i++) {
                box.visibility_arr[i].Set_phi(box.grid3d->phi);
                box.visibility_arr[i].Calculate_all();
            }
#pragma omp barrier
            velocity_calculator->Set_phi(grid->phi);
            velocity_calculator->Set_psi(box.visibility_arr);
            grid->Update_velocity(velocity_calculator);
#endif
        }
        cout << "iteration time cost" << omp_get_wtime() - time_start << endl;
    }
    return MAX_ITERATION;
}

dtype Compute_delta(const Grid3d *grid, IdxType i, IdxType j, IdxType k) {
    auto idx_ijk = grid->Index(i, j, k);
    // central difference of norm gradient for curvature driven part
    dtype phi_x = (grid->phi[grid->Index(i + 1, j, k)] - grid->phi[grid->Index(i - 1, j, k)]) / 2;
    dtype phi_y = (grid->phi[grid->Index(i, j + 1, k)] - grid->phi[grid->Index(i, j - 1, k)]) / 2;
    //// issue1: I need to increase the boundary distance since here it happened that (i, j, k - 1) is OUTSIDE
    //// fixed
    dtype phi_z = (grid->phi[grid->Index(i, j, k + 1)] - grid->phi[grid->Index(i, j, k - 1)]) / 2;
    if (isnan(phi_x) || isnan(phi_y) || isnan(phi_z)) {
        cerr << "nan_here" << endl;
        exit(1);
    }
    dtype val = 0;
#if USE_SIL
    if (grid->Phi[idx_ijk] == 0) return 0;
    else {
#endif
        if (phi_x != 0 || phi_y != 0 || phi_z != 0) {
            //part 1: Phi*div(unit normal) * \norm{\nabla phi}
            dtype phi_xx =
                    grid->phi[grid->Index(i + 1, j, k)] + grid->phi[grid->Index(i - 1, j, k)] - 2 * grid->phi[idx_ijk];
            dtype phi_yy =
                    grid->phi[grid->Index(i, j + 1, k)] + grid->phi[grid->Index(i, j - 1, k)] - 2 * grid->phi[idx_ijk];
            dtype phi_zz =
                    grid->phi[grid->Index(i, j, k + 1)] + grid->phi[grid->Index(i, j, k - 1)] - 2 * grid->phi[idx_ijk];
            assert(grid->isValidRange(i + 1, j + 1, k) && grid->isValidRange(i + 1, j - 1, k) &&
                   grid->isValidRange(i - 1, j + 1, k) && grid->isValidRange(i - 1, j - 1, k));
            dtype phi_xy = (grid->phi[grid->Index(i + 1, j + 1, k)] - grid->phi[grid->Index(i - 1, j + 1, k)]
                            - grid->phi[grid->Index(i + 1, j - 1, k)] + grid->phi[grid->Index(i - 1, j - 1, k)]) / 4;
            assert(grid->isValidRange(i, j + 1, k + 1) && grid->isValidRange(i, j + 1, k - 1) &&
                   grid->isValidRange(i, j - 1, k + 1) && grid->isValidRange(i, j - 1, k - 1));
            dtype phi_yz = (grid->phi[grid->Index(i, j + 1, k + 1)] - grid->phi[grid->Index(i, j - 1, k + 1)]
                            - grid->phi[grid->Index(i, j + 1, k - 1)] + grid->phi[grid->Index(i, j - 1, k - 1)]) / 4;
            assert(grid->isValidRange(i + 1, j, k + 1) && grid->isValidRange(i + 1, j, k - 1) &&
                   grid->isValidRange(i - 1, j, k + 1) && grid->isValidRange(i - 1, j, k - 1));
            dtype phi_xz = (grid->phi[grid->Index(i + 1, j, k + 1)] - grid->phi[grid->Index(i - 1, j, k + 1)]
                            - grid->phi[grid->Index(i + 1, j, k - 1)] + grid->phi[grid->Index(i - 1, j, k - 1)]) / 4;
            val += grid->Phi[idx_ijk] * ((pow(phi_y, 2) + pow(phi_z, 2)) * phi_xx
                                         + (pow(phi_x, 2) + pow(phi_z, 2)) * phi_yy
                                         + (pow(phi_x, 2) + pow(phi_y, 2)) * phi_zz
                                         - 2 * phi_x * phi_y * phi_xy
                                         - 2 * phi_x * phi_z * phi_xz
                                         - 2 * phi_y * phi_z * phi_yz)
                   / (phi_x * phi_x + phi_y * phi_y + phi_z * phi_z);
//                                    grid->Phi[idx_ijk] = val;

        } else {
////                                    grid->Phi[idx_ijk] = 0;
            val += 0;
        }
#if USE_SIL
    }
#endif
#if USE_SIL

#else
    // part 2: <\nabla_x Phi, unit normal> * \norm{\nabla phi}
    // we use upwind scheme to approximate each entry of the inner product.
    // and use central difference to approximate \nabla Phi
    assert(grid->isValidRange(i + 1, j, k) && grid->isValidRange(i - 1, j, k));
    dtype Phi_x = (grid->Phi[grid->Index(i + 1, j, k)] - grid->Phi[grid->Index(i - 1, j, k)]) / 2;
    assert(grid->isValidRange(i, j + 1, k) && grid->isValidRange(i, j - 1, k));
    dtype Phi_y = (grid->Phi[grid->Index(i, j + 1, k)] - grid->Phi[grid->Index(i, j - 1, k)]) / 2;
    assert(grid->isValidRange(i, j, k + 1) && grid->isValidRange(i, j, k - 1));
    dtype Phi_z = (grid->Phi[grid->Index(i, j, k + 1)] - grid->Phi[grid->Index(i, j, k - 1)]) / 2;
    // important part that needs to consider upwind scheme
    phi_x = Phi_x >= 0 ? grid->phi[grid->Index(i + 1, j, k)] - grid->phi[grid->Index(i, j, k)] : grid->phi[grid->Index(i, j, k)] - grid->phi[grid->Index(i - 1, j, k)];
    phi_y = Phi_y >= 0 ? grid->phi[grid->Index(i, j + 1, k)] - grid->phi[grid->Index(i, j, k)] : grid->phi[grid->Index(i, j, k)] - grid->phi[grid->Index(i, j - 1, k)];
    phi_z = Phi_z >= 0 ? grid->phi[grid->Index(i, j, k + 1)] - grid->phi[grid->Index(i, j, k)] : grid->phi[grid->Index(i, j, k)] - grid->phi[grid->Index(i, j, k - 1)];
    if (phi_x !=0 || phi_y != 0 || phi_z != 0) {
        val += Phi_x * phi_x + Phi_y * phi_y + Phi_z * phi_z;
        assert(!isnan(val));
    }
    else {
        val += 0;
    }
#endif
    return val;
}

