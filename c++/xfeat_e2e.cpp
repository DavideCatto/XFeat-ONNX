#pragma once

#include "xfeat_e2e.h"
#include "utils.h"

int xFeatE2EModule::initOrtEnv(Configuration cfg)
{
    DEBUG_MSG("< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->");
    try
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "XFeat Extractor and Matcher");

        // Create Session Option Extraction
        sess_opt = Ort::SessionOptions();
        //std::cout << std::thread::hardware_concurrency() << "Threads found" std::endl;
        sess_opt.SetInterOpNumThreads(std::thread::hardware_concurrency());
        sess_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sess_opt.SetLogSeverityLevel(3);

        // Check Device
        if (cfg.device == "cuda") {
            DEBUG_MSG("[INFO] OrtSessionOptions Append CUDAExecutionProvider");
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1;
            cuda_options.do_copy_in_default_stream = 1;
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            sess_opt.AppendExecutionProvider_CUDA(cuda_options);
            sess_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }

#if _WIN32
        DEBUG_MSG("[INFO] Env _WIN32 change modelpath from multi byte to wide char ...");
        const wchar_t* modelPath = multi_Byte_To_Wide_Char(cfg.xfeatPath);
#else
        const char* modelPath = cfg.xfeatPath;
#endif // _WIN32


        // ---------------------------------------------------------
        //                 Create Session
        // ---------------------------------------------------------
        sess = std::make_unique<Ort::Session>(env, modelPath, sess_opt);
        size_t numInputNodes = sess->GetInputCount();
        inNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            inNodeNames.emplace_back(_strdup(sess->GetInputNameAllocated(i, allocator).get()));
            inNodeShapes.emplace_back(sess->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        size_t numOutputNodes = sess->GetOutputCount();
        outNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            outNodeNames.emplace_back(_strdup(sess->GetOutputNameAllocated(i, allocator).get()));
            outNodeShapes.emplace_back(sess->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        DEBUG_MSG("[INFO] ONNXRuntime environment created successfully.");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}



int xFeatE2EModule::extractKptsAndMatchInference(Configuration cfg, const cv::Mat& img_ref, const cv::Mat& img_curr)
{
    DEBUG_MSG("< - * -------- Extractor Inference START -------- * ->");
    try
    {
        // Dynamic InputNodeShapes is [1,3,-1,-1] or [1,1,-1,-1]
        DEBUG_MSG("[INFO] Image Reference Size : " << img_ref.size() << " Channels : " << img_ref.channels());
        DEBUG_MSG("[INFO] Image Current Size : " << img_curr.size() << " Channels : " << img_curr.channels());

        // Build src input node shape and destImage input node shape
        int srcInputTensorSize, destInputTensorSize;
        inNodeShapes[0] = { 1 , 3 , img_ref.size().height , img_ref.size().width };
        inNodeShapes[1] = { 1 , 3 , img_curr.size().height , img_curr.size().width };
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        // Prepare Input Tensor
        std::vector<Ort::Value> inputTensor;        // Onnxruntime allowed input

        // Normalize Image and Swap BCHW
        cv::Mat blob_img_ref = cv::dnn::blobFromImage(img_ref, 1 / 255.0, cv::Size(img_ref.size().width, img_ref.size().height), (0, 0, 0), false, false);
        cv::Mat blob_img_curr = cv::dnn::blobFromImage(img_curr, 1 / 255.0, cv::Size(img_curr.size().width, img_curr.size().height), (0, 0, 0), false, false);
        size_t in_tensor_size_ref = blob_img_ref.total();
        size_t in_tensor_size_curr = blob_img_curr.total();

        inputTensor.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, (float*)blob_img_ref.data, in_tensor_size_ref, inNodeShapes[0].data(), inNodeShapes[0].size()));

        inputTensor.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, (float*)blob_img_curr.data, in_tensor_size_curr, inNodeShapes[1].data(), inNodeShapes[1].size()));


        // Start Inference
        auto time_start = std::chrono::high_resolution_clock::now();
        auto outputTensor = sess->Run(Ort::RunOptions{ nullptr },
            inNodeNames.data(),
            inputTensor.data(),
            inputTensor.size(),
            outNodeNames.data(),
            outNodeNames.size());
        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        //time_extr_match += diff;
        // End extract Keypoints


        for (auto& tensor : outputTensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }

        // Move out tensor
        out_tensors = std::move(outputTensor);

        DEBUG_MSG("[INFO] Xfeat Extractor and Matching inference finish ...");
        std::cout << "[INFO] Xfeat Extractor and Matching inference time : " << diff << " [ms]" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] Xfeat Extractor inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


// -----------------------------------------------------
//                  MATCH FUNCTIONS
// -----------------------------------------------------

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> xFeatE2EModule::extractKeypointsAndMatch(Configuration cfg, const cv::Mat& imgRef, const cv::Mat& imgCurr)
{
    DEBUG_MSG("< - * -------- INFERENCE IMAGE START -------- * ->");

    if (imgRef.empty() || imgCurr.empty())
    {
        throw  "[ERROR] ImageEmptyError ";
    }
    cv::Mat imgRefRGB, imgCurrRGB;

    // Clear kpts!
    clearKeypointsMatch();

    // Convert BGR2RGB Image
    cv::cvtColor(imgRef, imgRefRGB, cv::COLOR_BGR2RGB);
    cv::cvtColor(imgCurr, imgCurrRGB, cv::COLOR_BGR2RGB);

    // Extract Keypoints From Images
    auto res = extractKptsAndMatchInference(cfg, imgRefRGB, imgCurrRGB);

    // Post Process Keypoints Matching
    if (res == EXIT_SUCCESS)
        xFeatE2EModule::matchKptsPostProcess(cfg);

    // Clear Tensors
    out_tensors.clear();


    // Return Keypoints match
    return kpts_match;
}



int xFeatE2EModule::matchKptsPostProcess(Configuration cfg)
{
    try {
        std::vector<int64_t> matches0_Shape = out_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* mkpts_0_ptr = (int64_t*)out_tensors[0].GetTensorMutableData<void>();
        //printf("[RESULT INFO] kpts0 Shape : (%lld , %lld , %lld)\n", kpts0_Shape[0], kpts0_Shape[1], kpts0_Shape[2]);
        DEBUG_MSG("[RESULT INFO] matches0 Shape : (" + std::to_string(matches0_Shape[0]) + " , " +
            std::to_string(matches0_Shape[1]) + ")");

        std::vector<int64_t> matches1_Shape = out_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* mkpts_1_ptr = (int64_t*)out_tensors[1].GetTensorMutableData<void>();
        //printf("[RESULT INFO] kpts1 Shape : (%lld , %lld , %lld)\n", kpts1_Shape[0], kpts1_Shape[1], kpts1_Shape[2]);
        DEBUG_MSG("[RESULT INFO] matches1 Shape : (" + std::to_string(matches1_Shape[0]) + " , " +
            std::to_string(matches1_Shape[1]) + ")");

        // Process kpts0 and kpts1
        std::vector<cv::Point2f> kpts0_f, kpts1_f;
        std::vector<float> match_scores;

        auto mkpts_0 = cv::Mat(matches0_Shape[0], matches0_Shape[1], CV_32F, mkpts_0_ptr).clone();
        auto mkpts_1 = cv::Mat(matches1_Shape[0], matches1_Shape[1], CV_32F, mkpts_1_ptr).clone();


        for (int i = 0; i < matches0_Shape[0]; i++)
        {
            kpts0_f.emplace_back(cv::Point2f(mkpts_0.at<float>(i, 0), mkpts_0.at<float>(i, 1)));
            kpts1_f.emplace_back(cv::Point2f(mkpts_1.at<float>(i, 0), mkpts_1.at<float>(i, 1)));
        }

        kpts_match.first = kpts0_f;
        kpts_match.second = kpts1_f;

        DEBUG_MSG("[INFO] Postprocessing operation completed successfully");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> xFeatE2EModule::getKeypointsMatch()
{
    return this->kpts_match;
}

void xFeatE2EModule::clearKeypointsMatch()
{
    kpts_match.first.clear();
    kpts_match.second.clear();
}

xFeatE2EModule::xFeatE2EModule(unsigned int threads) : num_threads(threads)
{
}

xFeatE2EModule::~xFeatE2EModule()
{
}
