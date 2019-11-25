//
// Created by himalaya on 9/14/19.
//

#include "filesio.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

vector<cv::String> SortFileNames(const string &folder_path)
{
    vector<cv::String> image_file_names;
    glob(folder_path, image_file_names);
    return image_file_names;
}

//void ReadImage(const string &folder_path, vector<Mat> &all_images)
//{
//    vector<cv::String> image_file_paths = SortFileNames(folder_path);
//    all_images.reserve(image_file_paths.size());
//    for (auto img_path : image_file_paths) {
//        all_images.emplace_back(imread(img_path));
//    }
//}

void Read_data(const string &parameter_file_path, const string &gray_img_folder_path, const string &seg_img_folder_path, vector<Camera> &all_cams)
{
    FileStorage fs(parameter_file_path, FileStorage::READ);
    Mat P;
    stringstream ss;
    string s;
    ////// this variable depends on how many cameras in the files.
    const int num_cams = 36;
    all_cams.reserve(num_cams);

    vector<cv::String> gray_img_file_paths = SortFileNames(gray_img_folder_path);
    if (gray_img_file_paths.empty() || gray_img_file_paths.size() != num_cams) {
        cerr << "source images folder is empty or the number of images is wrong!" << endl;
        exit(1);
    }
    vector<cv::String> seg_img_file_paths = SortFileNames(seg_img_folder_path);
    if (seg_img_file_paths.empty() || seg_img_file_paths.size() != num_cams) {
        cerr << "segmentation images folder is empty or the number of images is wrong!" << endl;
        exit(1);
    }

    for (int i = 0; i < num_cams; i++) {
        ss << setfill('0') << setw(3) << i;
        ss >> s;
        fs["viff" + s + "_matrix"] >> P;
        Mat K, R, t;
        decomposeProjectionMatrix(P, K, R, t);

        // normalize to homogeneous coordinate
        Mat t3 = Mat::zeros(3, 1, t.type());
        for (int j = 0; j < 3; j++) {
            t3.at<float>(j, 0) = t.at<float>(j, 0) / t.at<float>(3, 0);
        }

        Mat gray_img = imread(gray_img_file_paths[i], 0);
        Mat seg_img = imread(seg_img_file_paths[i], 0);
        Mat binary_img;
        threshold(seg_img, binary_img, 128, 255, THRESH_BINARY);
        all_cams.emplace_back(Camera(P, R, t3, K, gray_img, binary_img));
        ss.clear();
    }
}