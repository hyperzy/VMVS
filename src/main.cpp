#include "base.h"
#include "init.h"
#include "display.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <filesio.h>

using namespace cv;
using namespace std;
int main() {

    vector<Camera> all_cams;
    Read_data("../res/viff.xml", "../res/images", "../res/seg_images", all_cams);

    vector<Camera> all_cams1{all_cams[0], all_cams[4], all_cams[9], all_cams[13],
                             all_cams[18], all_cams[22], all_cams[27], all_cams[31]};
    all_cams.clear();
    all_cams.clear();
    all_cams.shrink_to_fit();

    for (auto &iter : all_cams1) {
        iter.Calculate_extrema();
    }
    Grid grid(all_cams1);
    grid.Init_grid();

    Show_3D(all_cams1, grid);


//    cout << all_cams[0].K << endl;
//    int size[3] = {2, 3, 4};
//    float *d1 = new float[24];
//    for (int i = 0; i < 24; i++) {
//        d1[i] = i;
//    }
//    Mat test(3, size, CV_32FC(1), d1);
//    cout << test.at<float>(1, 2, 2) << endl;
    return 0;
}