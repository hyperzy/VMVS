#include "base.h"
//#include "init.h"
#include "grid2d.h"
//#include "display.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <filesio.h>

#include <queue>

using namespace cv;
using namespace std;
int main() {
    double start, end;
    start = clock();

//    vector<Camera> all_cams;
//    Read_data("../res/viff.xml", "../res/images", "../res/seg_images", all_cams);
//
//    vector<Camera> all_cams1{all_cams[0], all_cams[4], all_cams[9], all_cams[13],
//                             all_cams[18], all_cams[22], all_cams[27], all_cams[31]};
//    all_cams.clear();
//    all_cams.clear();
//    all_cams.shrink_to_fit();
//
//    for (auto &iter : all_cams1) {
//        iter.Calculate_extrema();
//    }
//    Grid grid(all_cams1);
//    grid.Init_grid();
//
//    Show_3D(all_cams1, grid);
    Grid2d *grid = new Grid2d(40, 40);
    grid->FMM_init();
    auto new_grid = FMM2d(grid, true);
    new_grid->Approx_front();
    delete grid;
    Evolve(new_grid);
    end = clock();
    cout << (end - start) / CLOCKS_PER_SEC << endl;
//    cout << all_cams[0].K << endl;

    return 0;
}