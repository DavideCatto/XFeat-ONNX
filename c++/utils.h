#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <Windows.h>
#include <opencv2/opencv.hpp>

inline wchar_t* multi_Byte_To_Wide_Char(std::string& pKey)
{
    // string char*
    const char* pCStrKey = pKey.c_str();
    size_t pSize = MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, NULL, 0);
    wchar_t* pWCStrKey = new wchar_t[pSize];
    MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, pWCStrKey, pSize);
    return pWCStrKey;
}

// Keypoints Function
cv::Mat plotKeypoints(cv::Mat img, std::vector<cv::Point2f> kpts, int radius = 2, cv::Scalar color = cv::Scalar(255, 0, 0), int thickness = -1); 
cv::Mat plotKeypointsMatch(cv::Mat img_ref, cv::Mat img_curr, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> kpts_pair);
cv::Mat warpCornersAndDrawMatches(const std::vector<cv::Point2f>& refPoints, const std::vector<cv::Point2f>& dstPoints,
    const cv::Mat& img1, const cv::Mat& img2);
#endif // UTILS_H