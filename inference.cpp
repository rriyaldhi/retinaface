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
    std::vector<cv::Rect> rectangles = retinaFace.infer(value, s.width, s.height);
    std::cout << "Detected faces: " << rectangles.size() << std::endl;

    return 0;
}
