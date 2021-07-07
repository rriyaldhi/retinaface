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
    std::cout << rectangles.size() << std::endl;
    for (cv::Rect rectangle: rectangles)
    {
        std::cout << rectangle.x << " " << rectangle.y << " " << rectangle.width << " " << rectangle.height << std::endl;
    }

    return 0;
}
