//
// Created by himalaya on 9/14/19.
//

#ifndef VMVS_FILESIO_H
#define VMVS_FILESIO_H

#include "base.h"
#include <utility>
using namespace std;




/**
 * @folder_path is the directory to deal with
 *
 * return a vector consisting of all the paths of image.
 */
vector<cv::String> SortFileNames(const string &folder_path);


/** Read all the date required
 *
 * @param parameter_file_path The path of parameters file
 * @param folder_path A vector consisting of all the paths of images
 * @param all_cams Output of the function. It consists of all the camera parameters
 */
void Read_data(const string &parameter_file_path, const string &gray_img_folder_path, const string &seg_img_folder_path, vector<Camera> &all_cams);



#endif //VMVS_FILESIO_H
