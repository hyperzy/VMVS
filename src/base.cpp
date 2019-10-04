//
// Created by himalaya on 9/29/19.
//

#include "base.h"
#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

bool compare_x(Point &a1, Point &a2)
{
    return a1.x < a2.x;
}

bool compare_y(Point &a1, Point &a2)
{
    return a1.y < a2.y;
}

void Camera::Calculate_extrema()
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(seg_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    if (contours[0].empty()) {
        cerr << "No contours extracted!" << endl;
    }
    sort(contours[0].begin(), contours[0].end(), compare_x);

    leftmost = contours[0].front();
    rightmost = contours[0].back();

    sort(contours[0].begin(), contours[0].end(), compare_y);
    topmost = contours[0].front();
    bottommost = contours[0].back();

//    cout << seg_img.size() << endl;
//    cout << leftmost << endl;
//    cout << contours[0] << endl;
//    Mat test;
//    seg_img.copyTo(test);
//    for (int i = 0; i < 800; i++) {
//        test.at<uchar>(i, contours[0][0].x) = 128;
//    }
//    namedWindow("test", 0);
//    imshow("test", test);
//    waitKey(0);

}
