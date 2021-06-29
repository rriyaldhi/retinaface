#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include <dirent.h>
#include "NvInfer.h"
#include "decode.h"

#ifndef RETINAFACE_INFERENCE_H
#define RETINAFACE_INFERENCE_H

#define DEVICE 0
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4

using namespace nvinfer1;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

class RetinaFace
{
private:
    static const int INPUT_H = decodeplugin::INPUT_H;
    static const int INPUT_W = decodeplugin::INPUT_W;
    static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;

    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void initializeEngine();

    static void doInference(IExecutionContext* context, float* input, float* output, int batchSize);
    static inline cv::Mat preprocess(cv::Mat& img, int input_w, int input_h);
    static cv::Rect getRectangles(cv::Mat& img, int input_w, int input_h, float *bbox, float *lmk);
    static float iou(float lbox[4], float rbox[4]);
    static bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b);
    static inline void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh);
public:
    RetinaFace();
    std::vector<cv::Rect> infer(std::vector<uint8_t> value, uint32_t width, uint32_t height);
    ~RetinaFace();
};

#endif
