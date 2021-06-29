#include <iostream>
#include <vector>
#include "retinaface.h"

int main(int argc, char** argv) {
    RetinaFace retinaFace = RetinaFace();
    cv::Mat imageRgb;
    cv::Mat imageBgr = cv::imread("../images/1.360.jpeg");
//    cv::cvtColor(imageBgr, imageRgb, cv::COLOR_BGR2RGB);

    std::vector<uint8_t> value;
    if (imageBgr.isContinuous()) {
        value.assign(imageBgr.data, imageBgr.data + imageBgr.total() * imageBgr.channels());
    }

    std::vector<cv::Rect> rectangles = retinaFace.infer(value, 360, 246);
    std::cout << rectangles.size() << std::endl;
    for (cv::Rect rectangle: rectangles)
    {
        std::cout << rectangle.x << " " << rectangle.y << " " << rectangle.width << " " << rectangle.height << std::endl;
    }

    return 0;
}
