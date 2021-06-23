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

    Logger logger;
    this->runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    this->context = this->engine->createExecutionContext();
    assert(context != nullptr);
}

void RetinaFace::doInference(IExecutionContext* context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(RetinaFace::INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(RetinaFace::OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * RetinaFace::INPUT_H * RetinaFace::INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * RetinaFace::OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * RetinaFace::INPUT_H * RetinaFace::INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * RetinaFace::OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

std::vector<cv::Rect> RetinaFace::infer(std::string imagePath) {
    cv::Mat img = cv::imread(imagePath);

    cv::Mat pr_img = RetinaFace::preprocess(img, RetinaFace::INPUT_W, RetinaFace::INPUT_H);
    static float data[3 * RetinaFace::INPUT_H * RetinaFace::INPUT_W];
    float *p_data = &data[0];
    for (int i = 0; i < RetinaFace::INPUT_H * RetinaFace::INPUT_W; i++) {
        p_data[i] = pr_img.at<cv::Vec3b>(i)[0];
        p_data[i + RetinaFace::INPUT_H * RetinaFace::INPUT_W] = pr_img.at<cv::Vec3b>(i)[1];
        p_data[i + 2 * RetinaFace::INPUT_H * RetinaFace::INPUT_W] = pr_img.at<cv::Vec3b>(i)[2];
    }

    static float prob[RetinaFace::OUTPUT_SIZE];
    std::cout << "inferencing" << std::endl;
    this->doInference(this->context, data, prob, 1);

    std::cout << "postprocessing" << std::endl;
    std::vector<decodeplugin::Detection> res;
    nms(res, &prob[0], IOU_THRESH);
    cv::Mat tmp = img.clone();
    std::vector<cv::Rect> rectangles;
    for (size_t j = 0; j < res.size(); j++) {
        if (res[j].class_confidence < CONF_THRESH) continue;
        cv::Rect rectangle = get_rect_adapt_landmark(tmp, RetinaFace::INPUT_W, RetinaFace::INPUT_H, res[j].bbox, res[j].landmark);
        rectangles.push_back(rectangle);
    }

    return rectangles;
}

RetinaFace::~RetinaFace() {
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

int RetinaFace::preprocess(cv::Mat &img, int input_w, int input_h) {
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

