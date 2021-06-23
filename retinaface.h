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

static inline cv::Rect get_rect_adapt_landmark(cv::Mat& img, int input_w, int input_h, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (input_h - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (input_h - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w;
        }
    } else {
        l = (bbox[0] - (input_w - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (input_w - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (input_w - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
        }
    }
    return cv::Rect(l, t, r-l, b-t);
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

    static cv::Mat preprocess(cv::Mat& img, int input_w, int input_h);
    static float iou(float* lbox, float* rbox);
    static bool cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b);
    static void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4);
public:
    RetinaFace();
    std::vector<cv::Rect> infer(std::string imagePath);
    ~RetinaFace();
};

#endif
