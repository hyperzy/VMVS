//
// Created by himalaya on 9/27/19.
//

#ifndef VMVS_BASE_H
#define VMVS_BASE_H


#include <opencv2/core.hpp>
#include <vector>

typedef float dtype;
typedef cv::Point3f Point3;
typedef cv::Vec3f Vec3;
typedef cv::Vec2f Vec2;
#define DTYPE CV_32F
constexpr dtype INF = 5e10;
typedef unsigned short IdxType;
typedef unsigned short DimUnit;

class Camera
{
public:
    const cv::Mat P, R, K, t, gray_img, seg_img;
    // two points of each vector;
    cv::Point leftmost;
    cv::Point rightmost;
    cv::Point topmost;
    cv::Point bottommost;
    Camera(cv::Mat P, cv::Mat R, cv::Mat t, cv::Mat K, cv::Mat gray_img, cv::Mat seg_img):
            P(std::move(P)), R(std::move(R)), t(std::move(t)), K(std::move(K)),
            gray_img(std::move(gray_img)), seg_img(std::move(seg_img))
    {
    }

    /**
     *  Computing the leftmost, right most, top most, bottom most coordinate on the image
     */
    void Calculate_extrema();

};




#endif //VMVS_BASE_H
