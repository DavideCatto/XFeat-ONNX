#include <iostream>
#include "xfeat_e2e.h"
#include "utils.h"
#include "configuration.h"

int main()
{
    // Variables
    bool isDense = true;
    std::string path_xfeat = "PATH TO XFeat";
    std::string path_models = path_xfeat + "weights/";
    std::string path_imgs = path_xfeat + "assets/";
    std::string fname_xfeat_match = "xfeat_e2e.onnx";
    bool SHOW = true;
    int num_test = 20;

    // Change fname model
    if (isDense) {
        fname_xfeat_match = "xfeat_dense_e2e.onnx";
    }

    // Configuration
    Configuration cfg;
    cfg.xfeatPath = path_models + "/" + fname_xfeat_match;
    cfg.device = "cuda"; // Support "cpu" / "cuda"
    cfg.isDense = isDense;
    cfg.show = SHOW;

    // Create XFeat Module
    xFeatE2EModule* xFeatModel = new xFeatE2EModule();
    xFeatModel->initOrtEnv(cfg);

    // Load Images
    cv::Mat img_ref = cv::imread(path_imgs + "ref.png");
    cv::Mat img_curr = cv::imread(path_imgs + "tgt.png");

    // Extract Keypoints
    if (false) {
        // Extract Keypoints and plot on image
        auto startTime = std::chrono::steady_clock::now();
        auto kpts_matched = xFeatModel->extractKeypointsAndMatch(cfg, img_ref, img_curr);
        auto endTime = std::chrono::steady_clock::now();

        cv::Mat img_ref_draw = plotKeypoints(img_ref, kpts_matched.first);
        cv::Mat img_curr_draw = plotKeypoints(img_curr, kpts_matched.second);

        if (SHOW) {
            cv::imshow("Reference", img_ref_draw);
            cv::imshow("Current", img_curr_draw);
            cv::waitKey();
        }

        // Save Image
        cv::imwrite(path_models + "Results/ref_c++.png", img_ref_draw);
        cv::imwrite(path_models + "Results/curr_c++.png", img_curr_draw);
    }
    else { // Extract Keypoints and match results
        auto kpts_matches = xFeatModel->extractKeypointsAndMatch(cfg, img_ref, img_curr);

        // Plot Keypoints
        cv::Mat figure = warpCornersAndDrawMatches(kpts_matches.first, kpts_matches.second, img_ref, img_curr);
        if (SHOW) {
            cv::imshow("Matches", figure);
            cv::waitKey();
        }
        cv::imwrite(path_imgs + "match_e2e_c++.png", figure);
    }
}

