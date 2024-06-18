#include <iostream>
#include "xfeat.h"
#include "utils.h"
#include "configuration.h"

int main()
{
    // Variables
	bool isDense = true;
    std::string path_xfeat = "PATH TO XFeat";
    std::string path_models = path_xfeat + "weights/";
    std::string path_imgs = path_xfeat + "assets/";
    std::string fname_xfeat = "xfeat.onnx";
    std::string fname_matcher = "matching.onnx";
    bool SHOW = true;

    // Change fname model
	if (isDense) {
		fname_xfeat = "xfeat_dense.onnx";
        fname_matcher = "matching_dense.onnx";
	}


    // Configuration
    Configuration cfg;
    cfg.xfeatPath = path_models  + fname_xfeat;
    cfg.matcherPath = path_models  + fname_matcher;
    cfg.device = "cuda"; // Support "cpu" / "cuda"
	cfg.isDense = isDense;
    cfg.show = SHOW;

    // Create XFeat Module
    xFeatModule *xFeatModel = new xFeatModule();
    xFeatModel->initOrtEnv(cfg);
    
    // Load Images
    cv::Mat img_ref = cv::imread(path_imgs + "ref.png");
    cv::Mat img_curr = cv::imread(path_imgs + "tgt.png");

    // Compare 

    // Extract Keypoints
    if (false) {
        // Extract Keypoints and plot on image
        auto startTime = std::chrono::steady_clock::now();
        auto kpts_ref = xFeatModel->extractKeypoints(cfg, img_ref);
        auto kpts_curr = xFeatModel->extractKeypoints(cfg, img_curr);
        auto endTime = std::chrono::steady_clock::now();

        cv::Mat img_ref_draw = plotKeypoints(img_ref, kpts_ref);
        cv::Mat img_curr_draw = plotKeypoints(img_curr, kpts_curr);

        if (SHOW) {
            cv::imshow("Reference", img_ref_draw);
            cv::imshow("Current", img_curr_draw);
            cv::waitKey();
        }

        // Save Image
        cv::imwrite(path_imgs + "ref_res_c++.png", img_ref_draw);
        cv::imwrite(path_imgs + "tgt_c++.png", img_curr_draw);
    }
    else { // Extract Keypoints and match results
        auto kpts_matches = xFeatModel->extractKeypointsAndMatch(cfg, img_ref, img_curr);

        // Plot Keypoints
        cv::Mat figure = warpCornersAndDrawMatches(kpts_matches.first, kpts_matches.second, img_ref, img_curr);
        if (SHOW) {
            cv::imshow("Matches", figure);
            cv::waitKey();
        }
        cv::imwrite(path_imgs + "match_c++.png", figure);
    }  
}

