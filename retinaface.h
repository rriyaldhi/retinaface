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
#include "common.hpp"

#ifndef RETINAFACE_INFERENCE_H
#define RETINAFACE_INFERENCE_H

#define DEVICE 0
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4

static const int INPUT_H = decodeplugin::INPUT_H;
static const int INPUT_W = decodeplugin::INPUT_W;
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

class RetinaFace
{
private:
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
