#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define DEVICE 0  // GPU id
#define CONF_THRESH 0.75
#define IOU_THRESH 0.4

// stuff we know about the network and the input/output blobs
static const int INPUT_H = decodeplugin::INPUT_H;  // H, W must be able to  be divided by 32.
static const int INPUT_W = decodeplugin::INPUT_W;;
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * 15 + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static Logger gLogger;

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file("retina_r50.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // prepare input data ---------------------------
    static float data[3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;

    cv::Mat img = cv::imread("../images/2.4k.jpeg");
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
    //cv::imwrite("preprocessed.jpg", pr_img);

    // For multi-batch, I feed the same image multiple times.
    // If you want to process different images in a batch, you need adapt it.
    float *p_data = &data[0];
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
        p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
        p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    //ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    static float prob[OUTPUT_SIZE];
    int total = 0;
    int n = 1001;
    for (int cc = 0; cc < n; cc++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
        if (cc > 0)
            total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << (total / (n - 1)) << std::endl;

    std::vector<decodeplugin::Detection> res;
    nms(res, &prob[0], IOU_THRESH);
    std::cout << "number of detections -> " << prob[0] << std::endl;
    std::cout << " -> " << prob[10] << std::endl;
    std::cout << "after nms -> " << res.size() << std::endl;
    cv::Mat tmp = img.clone();
    for (size_t j = 0; j < res.size(); j++) {
        if (res[j].class_confidence < CONF_THRESH) continue;
        cv::Rect r = get_rect_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
        cv::rectangle(tmp, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        //cv::putText(tmp, std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        for (int k = 0; k < 10; k += 2) {
            cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
        }
    }
    cv::imwrite("_result.jpg", tmp);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
