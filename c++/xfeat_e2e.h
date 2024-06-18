#pragma once
#ifndef XFEAT_E2E_MODULE_H
#define XFEAT_E2E_MODULE_H

#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include "configuration.h"


class xFeatE2EModule
{
private:
    const unsigned int num_threads;

    // Onnx Stuff
    Ort::Env env;
    Ort::SessionOptions sess_opt;
    std::unique_ptr<Ort::Session> sess;
    Ort::AllocatorWithDefaultOptions allocator;

    // Input Node
    std::vector<char*> inNodeNames;
    std::vector<std::vector<int64_t>> inNodeShapes;

    // Output Node
    std::vector<char*> outNodeNames;
    std::vector<std::vector<int64_t>> outNodeShapes;

    std::vector<Ort::Value> out_tensors;
    // End Onnx Stuff

    // Model Config
    float matchThresh = 0.0f;

    // Debug
    long long time_ext_kpts = 0.0f;
    long long time_match_kpts = 0.0f;

    // Keypoints Results
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> kpts_match;

private:

    // Extract Keypoints Inference Model
    int extractKptsInference(Configuration cfg, const cv::Mat& image);
    std::pair<std::vector<cv::Point2f>, float*> kptsPostProcess(Configuration cfg, std::vector<Ort::Value> tensor);

    // Match Keypoints
    int matchKptsInference();

    //std::vector<cv::Point2f> kptsMatchPreProcess(std::vector<cv::Point2f> kpts, int h, int w);
    //int matchKptsInference(Configuration cfg, std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float* desc0, float* desc1);
    int matchKptsPostProcess(Configuration cfg);
    void clearKeypointsMatch();

    int extractKptsAndMatchInference(Configuration cfg, const cv::Mat& img_ref, const cv::Mat& img_curr);

public:
    explicit xFeatE2EModule(unsigned int num_threads = 1);
    ~xFeatE2EModule();

    // Init Onnxruntime
    int initOrtEnv(Configuration cfg);

    // Match Images using Keypoints
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extractKeypointsAndMatch(Configuration cfg, const cv::Mat& imgRef, const cv::Mat& imgCurr);

    // Get Keypoints Match
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getKeypointsMatch();

};

#endif // XFEAT_E2E_MODULE_H