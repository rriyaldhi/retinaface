#include "retinaface.h"

RetinaFace::RetinaFace() {
    this->initializeEngine();
}

void RetinaFace::initializeEngine() {
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file("retinaface.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
}

void RetinaFace::doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
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

std::vector<cv::Rect> RetinaFace::infer(std::string imagePath) {
    cv::Mat img = cv::imread(imagePath);

    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
    static float data[3 * INPUT_H * INPUT_W];
    float *p_data = &data[0];
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        p_data[i] = pr_img.at<cv::Vec3b>(i)[0];
        p_data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1];
        p_data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[2];
    }

    static float prob[OUTPUT_SIZE];
    std::cout << "inferencing" << std::endl;
    this->doInference(*context, data, prob, 1);

    std::cout << "postprocessing" << std::endl;
    std::vector<decodeplugin::Detection> res;
    nms(res, &prob[0], IOU_THRESH);
    cv::Mat tmp = img.clone();
    std::vector<cv::Rect> rectangles;
    for (size_t j = 0; j < res.size(); j++) {
        if (res[j].class_confidence < CONF_THRESH) continue;
        cv::Rect rectangle = get_rect_adapt_landmark(tmp, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
        rectangles.push_back(rectangle)
    }

    return rectangles;
}

RetinaFace::~RetinaFace() {
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

