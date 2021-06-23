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

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

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

static inline void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[15 * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);
        //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
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
public:
    RetinaFace();
    std::vector<cv::Rect> infer(std::string imagePath);
    ~RetinaFace();
};

#endif
