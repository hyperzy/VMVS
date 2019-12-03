#include "base.h"
#include "init.h"
#include "display.h"
#include "evolution.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "filesio.h"
#include "../testvtk.cpp"

using namespace cv;
using namespace std;
int main() {
    double start, end;
    start = clock();
//    Show_3D("./final_data_s.stl");
    vector<Camera> all_cams;
    Read_data("../res/viff.xml", "../res/images", "../res/seg_images", all_cams);

//    vector<Camera> all_cams1{all_cams[0], all_cams[2], all_cams[4], all_cams[6], all_cams[9], all_cams[11], all_cams[13], all_cams[16],
//                             all_cams[18], all_cams[20], all_cams[22], all_cams[24], all_cams[27], all_cams[29], all_cams[31]};
    vector<Camera> all_cams1{all_cams[0], all_cams[4], all_cams[9], all_cams[13],
                             all_cams[18], all_cams[22], all_cams[27], all_cams[31]};
//    for (auto &iter : all_cams1) {
//        cout << iter.P << endl;
//    }
    all_cams.clear();
    all_cams.clear();
    all_cams.shrink_to_fit();
//    imshow("test", all_cams1[4].seg_img);
//    waitKey(0);
//    cout << endl;
//    for (auto &iter : all_cams1) {
//        cout << iter.P << endl;
//    }
    for (auto &iter : all_cams1) {
        iter.Calculate_extrema();
    }
    float radius = 7;
    float resolution = 1;
    radius = 7 / 1;
    BoundingBox box(all_cams1, 0.5);
    box.Init();
    // radius is not in use anymore
    Init_sphere_shape(box, 7);

    Show_3D(all_cams1, box);
#if USE_NEW
#else
    Evolve(box, all_cams1);
#endif
//    testvtk();
////2d test)
/*
    Grid2d *grid = new Grid2d(40, 40);
    grid->FMM_init();
    auto new_grid = FMM2d(grid, true);
    new_grid->Approx_front();
    delete grid;
    Evolve(new_grid);
    end = clock();
    cout << (end - start) / CLOCKS_PER_SEC << endl;
*/
//    cout << all_cams[0].K << endl;

    return 0;
}


//#include <vtkAutoInit.h>
//VTK_MODULE_INIT(vtkRenderingOpenGL2);
//VTK_MODULE_INIT(vtkInteractionStyle);
