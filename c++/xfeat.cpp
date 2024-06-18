#pragma once

#include "xfeat.h"
#include "utils.h"

// Constructor & Deconstructor
xFeatModule::xFeatModule(unsigned int threads) : num_threads(threads){}

xFeatModule::~xFeatModule(){
    env0.release();
    env1.release();
    sess_opt_extr.release();
    sess_opt_match.release();
}


// Onnx init
int xFeatModule::initOrtEnv(Configuration cfg)
{
    DEBUG_MSG("< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->");
    try
    {
        env0 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "XFeat Extractor");
        env1 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "XFeat Matcher");

        // Create Session Option Extraction
        sess_opt_extr = Ort::SessionOptions();
        sess_opt_extr.SetInterOpNumThreads(std::thread::hardware_concurrency());
        sess_opt_extr.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Create Session Option Match
        sess_opt_match = Ort::SessionOptions();
        sess_opt_match.SetInterOpNumThreads(std::thread::hardware_concurrency());
        sess_opt_match.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        sess_opt_extr.SetLogSeverityLevel(3);
        sess_opt_match.SetLogSeverityLevel(3);

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

            sess_opt_extr.AppendExecutionProvider_CUDA(cuda_options);
            sess_opt_extr.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            sess_opt_match.AppendExecutionProvider_CUDA(cuda_options);
            sess_opt_match.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }

#if _WIN32
        DEBUG_MSG("[INFO] Env _WIN32 change modelpath from multi byte to wide char ...");
        const wchar_t* extractor_modelPath = multi_Byte_To_Wide_Char(cfg.xfeatPath);
        const wchar_t* matcher_modelPath = multi_Byte_To_Wide_Char(cfg.matcherPath);
#else
        const char* extractor_modelPath = cfg.xfeatPath;
        const char* matcher_modelPath = cfg.matcherPath;
#endif // _WIN32


        // ---------------------------------------------------------
        //                 Create Extraction Session
        // ---------------------------------------------------------
        sessExtract = std::make_unique<Ort::Session>(env0, extractor_modelPath, sess_opt_extr);
        size_t numInputNodes = sessExtract->GetInputCount();
        extInNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            extInNodeNames.emplace_back(_strdup(sessExtract->GetInputNameAllocated(i, allocator).get()));
            extInNodeShapes.emplace_back(sessExtract->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        size_t numOutputNodes = sessExtract->GetOutputCount();
        extOutNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            extOutNodeNames.emplace_back(_strdup(sessExtract->GetOutputNameAllocated(i, allocator).get()));
            extOutNodeShapes.emplace_back(sessExtract->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }


        // ---------------------------------------------------------
        //                 Create Match Session
        // ---------------------------------------------------------
        numInputNodes = 0;
        numOutputNodes = 0;

        sessMatch = std::make_unique<Ort::Session>(env1, matcher_modelPath, sess_opt_match);
        numInputNodes = sessMatch->GetInputCount();
        extInNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            matchInNodeNames.emplace_back(_strdup(sessMatch->GetInputNameAllocated(i, allocator).get()));
            matchInNodeShapes.emplace_back(sessMatch->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        numOutputNodes = sessMatch->GetOutputCount();
        extOutNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            matchOutNodeNames.emplace_back(_strdup(sessMatch->GetOutputNameAllocated(i, allocator).get()));
            matchOutNodeShapes.emplace_back(sessMatch->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
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

// Extract Keypoints
std::vector<cv::Point2f> xFeatModule::extractKeypoints(Configuration cfg, const cv::Mat& img)
{
    DEBUG_MSG("< - * -------- INFERENCE IMAGE START -------- * ->");

    if (img.empty())
    {
        throw  "[ERROR] ImageEmptyError ";
    }
    cv::Mat imgRGB;

    // Convert BGR2RGB Image
    cv::cvtColor(img, imgRGB, cv::COLOR_BGR2RGB);

    // Pre Process Image
    DEBUG_MSG("[INFO] => Pre-Process Image");
    //cv::Mat src = preProcessImage(cfg, imgRGB, scales[0]);

    // Extract Keypoints    
    extractKptsInference(cfg, imgRGB);

    // Post Process Keypoints Extraction
    auto tuple_kpts_desc = kptsPostProcess(cfg, std::move(ext_out_tensors[0]));

    // Rescale Points on image size
    //auto kpts_rescaled = kptsRescale(cfg, tuple_kpts_desc.first, scales[0]);

    // Clean
    ext_out_tensors.clear();

    return tuple_kpts_desc.first;
}

int xFeatModule::extractKptsInference(Configuration cfg, const cv::Mat& image)
{
    DEBUG_MSG("< - * -------- Extractor Inference START -------- * ->");
    try
    {
        // Dynamic InputNodeShapes is [1,3,-1,-1] or [1,1,-1,-1]
        DEBUG_MSG("[INFO] Image Size : " << image.size() << " Channels : " << image.channels());

        // Build src input node shape and destImage input node shape
        int srcInputTensorSize, destInputTensorSize;
        extInNodeShapes[0] = { 1 , 3 , image.size().height , image.size().width };
        
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        // Prepare Input Tensor
        std::vector<Ort::Value> inputTensor;        // Onnxruntime allowed input

        // Normalize Image and Swap BCHW
        cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(image.size().width, image.size().height), (0, 0, 0), false, false);
        size_t input_tensor_size = blob.total();
        inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, (float*)blob.data, input_tensor_size, extInNodeShapes[0].data(), extInNodeShapes[0].size()));

        // Start Inference
        auto time_start = std::chrono::high_resolution_clock::now();
        auto outputTensor = sessExtract->Run(Ort::RunOptions{ nullptr },
            extInNodeNames.data(),
            inputTensor.data(),
            inputTensor.size(),
            extOutNodeNames.data(),
            extOutNodeNames.size());
        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        time_ext_kpts += diff;
        // End extract Keypoints


        for (auto& tensor : outputTensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }

        // Move out tensor
        ext_out_tensors.emplace_back(std::move(outputTensor));

        DEBUG_MSG("[INFO] XFeatModule Extractor inference finish ...");
        std::cout << "[INFO] Extractor inference time : " << diff << " [ms]" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] XFeatModule Extractor inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, float*> xFeatModule::kptsPostProcess(Configuration cfg, std::vector<Ort::Value> tensor)
{
    std::pair<std::vector<cv::Point2f>, float*> tuple_kpts_desc;
    try {
        std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts = (int64_t*)tensor[0].GetTensorMutableData<void>();
        printf("[RESULT INFO] kpts Shape : (%lld , %lld)\n", kpts_Shape[0], kpts_Shape[1]);
        //DEBUG_MSG("[RESULT INFO] kpts Shape : (%lld , %lld , %lld)\n", kpts_Shape[0], kpts_Shape[1], kpts_Shape[2]);

        std::vector<int64_t> descriptors_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
        float* desc = (float*)tensor[1].GetTensorMutableData<void>();
        printf("[RESULT INFO] desc Shape : (%lld , %lld)\n", descriptors_Shape[0], descriptors_Shape[1]);
        //DEBUG_MSG("[RESULT INFO] desc Shape : (%lld , %lld , %lld)\n", descriptors_Shape[0], descriptors_Shape[1], descriptors_Shape[2]);

        std::vector<int64_t> score_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
        float* scores = (float*)tensor[2].GetTensorMutableData<void>();
        printf("[RESULT INFO] score Shape : (%lld)\n", score_Shape[0]);
        //DEBUG_MSG("[RESULT INFO] score Shape : (%lld , %lld)\n", score_Shape[0], score_Shape[1]);

        // Process kpts and descriptors
        std::vector<cv::Point2f> kpts_f;
        cv::Mat kptsMat = cv::Mat(kpts_Shape[0], kpts_Shape[1], CV_32F, kpts).clone();

        for (int i = 0; i < kpts_Shape[0]; i ++)
        {
            // Kpts Keypoints
            // kpts_f.emplace_back(cv::Point2f((kpts[i] + 0.5) / scales[0] - 0.5, (kpts[i + 1] + 0.5) / scales[0] - 0.5));
            kpts_f.emplace_back(cv::Point2f(kptsMat.at<float>(i, 0), kptsMat.at<float>(i, 1)));
        }

       
        // Add Kpts and descriptors
        tuple_kpts_desc.first = kpts_f;
        tuple_kpts_desc.second = desc;

        DEBUG_MSG("[INFO] Extractor postprocessing operation completed successfully");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] Extractor postprocess failed : " << ex.what() << std::endl;
    }

    return tuple_kpts_desc;
}


std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> xFeatModule::extractKeypointsAndMatch(Configuration cfg, const cv::Mat& imgRef, const cv::Mat& imgCurr)
{
    DEBUG_MSG("< - * -------- INFERENCE IMAGE START -------- * ->");

    if (imgRef.empty() || imgCurr.empty())
    {
        throw  "[ERROR] ImageEmptyError ";
    }
    cv::Mat imgRefRGB, imgCurrRGB;

    // Convert BGR2RGB Image
    cv::cvtColor(imgRef, imgRefRGB, cv::COLOR_BGR2RGB);
    cv::cvtColor(imgCurr, imgCurrRGB, cv::COLOR_BGR2RGB);

    // Extract Keypoints From Images
    extractKptsInference(cfg, imgRefRGB);
    extractKptsInference(cfg, imgCurrRGB);

    // Get Keypoints from Tensors
    matchKptsInference(cfg);

    // Post Process Keypoints Matching
    std::vector<cv::Point2f> kpts_ref, kpts_curr;
    matchKptsPostProcess();

    //auto kpts_ref_ext = kptsPostProcess(cfg, std::move(ext_out_tensors[0]));
    //auto kpts_curr_ext = kptsPostProcess(cfg, std::move(ext_out_tensors[1]));

    // Normalize Keypoints before match
    // auto kpts_ref_norm = kptsMatchPreProcess(kpts_ref_extracted.first, imgRefRGBPProc.rows, imgRefRGBPProc.cols);
    // auto kpts_curr_norm = kptsMatchPreProcess(kpts_curr_extracted.first, imgCurrRGBPProc.rows, imgCurrRGBPProc.cols);

    // Match Inference using kpts and descriptors
    //matchKptsInference(cfg, kpts_ref_ext, kpts_curr_ext, kpts_ref_ext.second, kpts_curr_ext.second);

    // Post Process Keypoints Matching
    //matchKptsPostProcess(cfg, kpts_ref_extracted.first, kpts_curr_extracted.first);

    // Clear Tensors
    ext_out_tensors.clear();
    match_out_tensors.clear();
    
    // Return Keypoints match
    return kpts_match;
}

int xFeatModule::matchKptsInference(Configuration cfg) {
    DEBUG_MSG("< - * -------- Matcher Inference START -------- * ->");
    try
    {
        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(ext_out_tensors[0][0]));
        input_tensors.push_back(std::move(ext_out_tensors[0][1]));
        input_tensors.push_back(std::move(ext_out_tensors[1][0]));
        input_tensors.push_back(std::move(ext_out_tensors[1][1]));
        
        // Add scales
        if (cfg.isDense) {
            input_tensors.push_back(std::move(ext_out_tensors[0][2]));
        }

        auto time_start = std::chrono::high_resolution_clock::now();
        auto output_tensor =
            sessMatch->Run(
                Ort::RunOptions{ nullptr },
                matchInNodeNames.data(),
                input_tensors.data(),
                input_tensors.size(),
                matchOutNodeNames.data(),
                matchOutNodeNames.size());

        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        time_match_kpts += diff;
        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        match_out_tensors = std::move(output_tensor);

        DEBUG_MSG("[INFO] lightGlueModule Matcher inference finish ...");
        std::cout << "[INFO] Matcher inference time : " << diff << " [ms]" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] XFeat Matcher inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int xFeatModule::matchKptsPostProcess()
{
    try {
        std::vector<int64_t> matches0_Shape = match_out_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* mkpts_0_ptr = (int64_t*)match_out_tensors[0].GetTensorMutableData<void>();
        printf("[RESULT INFO] matches0 Shape : (%lld , %lld)\n" , matches0_Shape[0] , matches0_Shape[1]);
        //DEBUG_MSG("[RESULT INFO] matches0 Shape : (%lld , %lld)\n", matches0_Shape[0], matches0_Shape[1]);

        std::vector<int64_t> matches1_Shape = match_out_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        float* mkpts_1_ptr = (float*)match_out_tensors[1].GetTensorMutableData<void>();
        printf("[RESULT INFO] matches1 Shape : (%lld)\n" , matches1_Shape[0]);
        //DEBUG_MSG("[RESULT INFO] mscores0 Shape : (%lld)\n", mscores0_Shape[0]);

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
