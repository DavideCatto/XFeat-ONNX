#pragma once
#include "utils.h"

// Plot Keypoints on Image
cv::Mat plotKeypoints(cv::Mat img, std::vector<cv::Point2f> kpts, int radius, cv::Scalar color, int thickness) {
    for (auto kpt : kpts) {
        cv::circle(img, cv::Point2f(kpt.x, kpt.y), radius, color, thickness);
    }
    return img;
}

// plot Keypoints Match
cv::Mat plotKeypointsMatch(cv::Mat img_ref, cv::Mat img_curr, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> kpts_pair) {

    // Create figure
    int h_figure = 450;

    // Image reference
    int x_offset = 0;
    cv::Mat img_ref_res, img_curr_res;

    // Resize Reference Image
    const cv::Mat& image = img_ref;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    auto w_img = image.cols;
    auto h_img = image.rows;
    float scale_h_ref = h_img / h_figure;
    cv::resize(image, img_ref_res, cv::Size(static_cast<int>(w_img / scale_h_ref), h_figure));
    cv::Rect roi_ref(cv::Point(0, 0), cv::Size(img_ref_res.cols, h_figure));
    x_offset = img_ref_res.cols;


    // Resize Current Image
    const cv::Mat& image_curr = img_curr;
    cv::cvtColor(image_curr, image_curr, cv::COLOR_BGR2RGB);
    w_img = image_curr.cols;
    h_img = image_curr.rows;
    float scale_h_curr = h_img / h_figure;
    cv::resize(image_curr, img_curr_res, cv::Size(static_cast<int>(w_img / scale_h_curr), h_figure));
    cv::Rect roi_curr = cv::Rect(cv::Point(x_offset, 0), cv::Size(img_curr_res.cols, img_curr_res.rows));

    // Add image to Figure
    cv::Size2f figureSize(x_offset + img_curr_res.cols, h_figure);
    cv::Mat figure(figureSize, CV_8UC3);
    img_ref_res.copyTo(figure(roi_ref));
    img_curr_res.copyTo(figure(roi_curr));

    // Assign keypoints 
    auto kpts0 = kpts_pair.first;
    auto kpts1 = kpts_pair.second;

    //plotMatches(figure, kpts0, kpts1, scale_h_ref, scale_h_curr, x_offset);

    cv::imshow("Figure", figure);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return figure;
}

cv::Mat warpCornersAndDrawMatches(const std::vector<cv::Point2f>& refPoints, const std::vector<cv::Point2f>& dstPoints,
    const cv::Mat& img1, const cv::Mat& img2)
{
    // Step 1: Calculate the Homography matrix and mask
    cv::Mat mask;
    cv::Mat H = cv::findHomography(refPoints, dstPoints, cv::USAC_MAGSAC, 3.5, mask, 1000, 0.999);
    mask = mask.reshape(1, mask.total());  // Flatten the mask

    // Step 2: Get corners of the first image (img1)
    std::vector<cv::Point2f> cornersImg1 = { cv::Point2f(0, 0), cv::Point2f(img1.cols - 1, 0),
                                            cv::Point2f(img1.cols - 1, img1.rows - 1), cv::Point2f(0, img1.rows - 1) };
    std::vector<cv::Point2f> warpedCorners(4);

    // Step 3: Warp corners to the second image (img2) space
    cv::perspectiveTransform(cornersImg1, warpedCorners, H);

    // Step 4: Draw the warped corners in image2
    cv::Mat img2WithCorners = img2.clone();
    for (size_t i = 0; i < warpedCorners.size(); i++) {
        cv::line(img2WithCorners, warpedCorners[i], warpedCorners[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }

    // Step 5: Prepare keypoints and matches for drawMatches function
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < refPoints.size(); i++) {
        if (mask.at<uchar>(i)) {  // Only consider inliers
            keypoints1.emplace_back(refPoints[i], 5);
            keypoints2.emplace_back(dstPoints[i], 5);
        }
    }
    for (size_t i = 0; i < keypoints1.size(); i++) {
        matches.emplace_back(i, i, 0);
    }

    // Draw inlier matches
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2WithCorners, keypoints2, matches, imgMatches, cv::Scalar(0, 255, 0), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return imgMatches;
}