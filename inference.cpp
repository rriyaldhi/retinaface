#include <iostream>
#include <vector>
#include "retinaface.h"

int main(int argc, char** argv) {
    RetinaFace retinaFace = RetinaFace();
    cv::Mat imageRgb;
    cv::Mat imageBgr = cv::imread(std::string(argv[1]));
    cv::cvtColor(imageBgr, imageRgb, cv::COLOR_BGR2RGB);

    std::vector<uint8_t> value;
    if (imageRgb.isContinuous()) {
        value.assign(imageRgb.data, imageRgb.data + imageRgb.total() * imageRgb.channels());
    }

    cv::Size s = imageRgb.size();
    int total = 0;
    int n = 1000;
    for (int i = 0; i < n; i++) {
        auto start = std::chrono::system_clock::now();
        std::vector<cv::Rect> rectangles = retinaFace.infer(value, s.width, s.height);
        auto end = std::chrono::system_clock::now();
        if (i > 0)
            total += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::vector<cv::Rect> rectangles = retinaFace.infer(value, s.width, s.height);
    std::cout << "Detected faces: " << rectangles.size() << std::endl;
    std::cout << "Total time: " << total / (n - 1) << " us" << std::endl;

    return 0;
}
