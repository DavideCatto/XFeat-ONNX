#pragma once
#ifndef XFEAT_MODULE_H
#define XFEAT_MODULE_H

#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include "configuration.h"


class xFeatModule
{
private:
    const unsigned int num_threads;

    // Onnx Stuff
    Ort::Env env0, env1;
    Ort::SessionOptions sess_opt_extr, sess_opt_match;
    std::unique_ptr<Ort::Session> sessExtract, sessMatch;
    Ort::AllocatorWithDefaultOptions allocator;

    // Extractor Keypoints
    std::vector<char*> extInNodeNames;
    std::vector<std::vector<int64_t>> extInNodeShapes;
    std::vector<char*> extOutNodeNames;
    std::vector<std::vector<int64_t>> extOutNodeShapes;

    // Matcher Keypoints
    std::vector<char*> matchInNodeNames;
    std::vector<std::vector<int64_t>> matchInNodeShapes;
    std::vector<char*> matchOutNodeNames;
    std::vector<std::vector<int64_t>> matchOutNodeShapes;

    std::vector<std::vector<Ort::Value>> ext_out_tensors;
    std::vector<Ort::Value> match_out_tensors;
    // End Onnx Stuff

    // Model Config
    float matchThresh = 0.0f;

    // Debug
    long long time_ext_kpts = 0.0f;
    long long time_match_kpts = 0.0f;

    // Keypoints Results
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> kpts_match;

private:
    

    // Preprocess Image
    //cv::Mat preProcessImage(Configuration cfg, const cv::Mat& srcImage, float& scale);

    // Extract Keypoints Inference Model
    int extractKptsInference(Configuration cfg, const cv::Mat& image);
    std::pair<std::vector<cv::Point2f>, float*> kptsPostProcess(Configuration cfg, std::vector<Ort::Value> tensor);



    //std::vector<cv::Point2f> kptsRescale(Configuration cfg, std::vector<cv::Point2f> kpts, float scale);

    // Match Keypoints
    int matchKptsInference(Configuration cfg);

    //std::vector<cv::Point2f> kptsMatchPreProcess(std::vector<cv::Point2f> kpts, int h, int w);
    //int matchKptsInference(Configuration cfg, std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float* desc0, float* desc1);
    int matchKptsPostProcess();

    /*

    void Extractor_PostProcessKeypoints(Configuration cfg, std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1);

    */
public:
    explicit xFeatModule(unsigned int num_threads = 1);
    ~xFeatModule();

    // Init Onnxruntime
    int initOrtEnv(Configuration cfg);

    // Extract Keypoints from image
    std::vector<cv::Point2f> extractKeypoints(Configuration cfg, const cv::Mat& img);

    //void setMatchThresh(float thresh);
    //double getTimer(std::string name);
    //float getMatchThresh();

    // Match Images using Keypoints
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extractKeypointsAndMatch(Configuration cfg, const cv::Mat& imgRef, const cv::Mat& imgCurr);

    // Get Keypoints Match
    //std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getKeypointsMatch();


};

#endif // XFEAT_MODULE_H