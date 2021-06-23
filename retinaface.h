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

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            std::max(lbox[0], rbox[0]), //left
            std::min(lbox[2], rbox[2]), //right
            std::max(lbox[1], rbox[1]), //top
            std::min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
}

static bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b) {
    return a.class_confidence > b.class_confidence;
}

class RetinaFace
{
private:
    static const int INPUT_H = decodeplugin::INPUT_H;
    static const int INPUT_W = decodeplugin::INPUT_W;
    static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";

    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void initializeEngine();
    void doInference(IExecutionContext* context, float* input, float* output, int batchSize);

    static inline cv::Mat preprocess(cv::Mat& img, int input_w, int input_h);
    static cv::Rect getRectangles(cv::Mat& img, int input_w, int input_h, float *bbox, float *lmk);
    static inline void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4);
public:
    RetinaFace();
    std::vector<cv::Rect> infer(std::string imagePath);
    ~RetinaFace();
};

#endif
