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

void RetinaFace:: doInference(IExecutionContext* context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("data");
    const int outputIndex = engine.getBindingIndex("prob");

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

std::vector<cv::Rect> RetinaFace::infer(std::vector<uint8_t> value, uint32_t width, uint32_t height) {
    cv::Mat imageRgb, imageBgr;
    imageRgb.create(height, width, CV_8UC3);
    std::copy(value.begin(), value.end(), imageRgb.data);
    cv::cvtColor(imageRgb, imageBgr, cv::COLOR_RGB2BGR);
    cv::Mat pr_img = RetinaFace::preprocess(imageBgr, RetinaFace::INPUT_W, RetinaFace::INPUT_H);
    static float data[3 * RetinaFace::INPUT_H * RetinaFace::INPUT_W];
    float *p_data = &data[0];
    for (int i = 0; i < RetinaFace::INPUT_H * RetinaFace::INPUT_W; i++) {
        p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
        p_data[i + RetinaFace::INPUT_H * RetinaFace::INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
        p_data[i + 2 * RetinaFace::INPUT_H * RetinaFace::INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
    }

    static float prob[RetinaFace::OUTPUT_SIZE];
    RetinaFace::doInference(this->context, data, prob, 1);

    std::vector<decodeplugin::Detection> res;
    RetinaFace::nms(res, &prob[0], IOU_THRESH);
    cv::Mat tmp = imageBgr.clone();
    std::vector<cv::Rect> rectangles;
    for (size_t j = 0; j < res.size(); j++) {
        if (res[j].class_confidence < CONF_THRESH) continue;
        cv::Rect rectangle = RetinaFace::getRectangles(tmp, RetinaFace::INPUT_W, RetinaFace::INPUT_H, res[j].bbox, res[j].landmark);
        rectangles.push_back(rectangle);
        cv::rectangle(tmp, rectangle, cv::Scalar(0x27, 0xC1, 0x36), 2);
        for (int k = 0; k < 10; k += 2) {
            cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
        }
    }
    cv::imwrite("result.jpg", tmp);

    return rectangles;
}

void RetinaFace::inferVideo(std::string input_video, std::string output_video) {
    cv::VideoCapture videoCapture(input_video);
    int width = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double FPS = videoCapture.get(cv::CAP_PROP_FPS);
    std::cout << width << std::endl;
    std::cout << height << std::endl;
    std::cout << FPS << std::endl;
    cv::VideoWriter videoWriter(output_video, cv::VideoWriter::fourcc('m','p', '4', 'v'), FPS, cv::Size(width, height));
    if (!videoWriter.isOpened())
    {
        std::cout << "ERROR: Failed to write the video" << std::endl;
    }
    while (true) {
        std::cout << "Reading..."<< std::endl;
        cv::Mat imageRgb;
        bool readSuccess = videoCapture.read(imageRgb);
        if (!readSuccess) {
            break;
        }
        cv::Mat imageBgr;
        cv::cvtColor(imageRgb, imageBgr, cv::COLOR_RGB2BGR);
        cv::Mat pr_img = RetinaFace::preprocess(imageBgr, RetinaFace::INPUT_W, RetinaFace::INPUT_H);
        static float data[3 * RetinaFace::INPUT_H * RetinaFace::INPUT_W];
        float *p_data = &data[0];
        for (int i = 0; i < RetinaFace::INPUT_H * RetinaFace::INPUT_W; i++) {
            p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
            p_data[i + RetinaFace::INPUT_H * RetinaFace::INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
            p_data[i + 2 * RetinaFace::INPUT_H * RetinaFace::INPUT_W] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
        }
        static float prob[RetinaFace::OUTPUT_SIZE];
        RetinaFace::doInference(this->context, data, prob, 1);

        std::vector<decodeplugin::Detection> res;
        RetinaFace::nms(res, &prob[0], IOU_THRESH);
        cv::Mat tmp = imageBgr.clone();
        std::vector<cv::Rect> rectangles;
        for (size_t j = 0; j < res.size(); j++) {
            if (res[j].class_confidence < CONF_THRESH) continue;
            cv::Rect rectangle = RetinaFace::getRectangles(tmp, RetinaFace::INPUT_W, RetinaFace::INPUT_H, res[j].bbox, res[j].landmark);
            rectangles.push_back(rectangle);
            cv::rectangle(tmp, rectangle, cv::Scalar(0x27, 0xC1, 0x36), 2);
            for (int k = 0; k < 10; k += 2) {
                cv::circle(tmp, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
            }
        }
        cv::cvtColor(tmp, imageRgb, cv::COLOR_BGR2RGB);
        std::cout << "Writing..." << std::endl;
        videoWriter.write(imageRgb);
    }
}

RetinaFace::~RetinaFace() {
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

cv::Mat RetinaFace::preprocess(cv::Mat &img, int input_w, int input_h) {
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

cv::Rect RetinaFace::getRectangles(cv::Mat& img, int input_w, int input_h, float *bbox, float *lmk) {
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

float RetinaFace::iou(float lbox[4], float rbox[4]) {
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

bool RetinaFace::cmp(const decodeplugin::Detection& a, const decodeplugin::Detection& b) {
    return a.class_confidence > b.class_confidence;
}

void RetinaFace::nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[15 * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), RetinaFace::cmp);
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);

        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (RetinaFace::iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
}