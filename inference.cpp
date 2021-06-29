#include <iostream>
#include <vector>
#include "retinaface.h"

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Argument is required!" << std::endl;
        std::cerr << "./inference /path/to/image" << std::endl;
        return -1;
    }

    RetinaFace retinaFace = RetinaFace();
    cv::Mat imageRgb, imageBgr;
    cv::Mat imageBgr = cv::imread(std::string(argv[1]));
    std::vector<uint8_t> value;
    if (imageBgr.isContinuous()) {
        value.assign(imageBgr.data, imageBgr.data + imageBgr.total() * imageBgr.channels());
    }
    cv::cvtColor(imageBgr, imageRgb, cv::COLOR_BGR2RGB);
    std::vector<cv::Rect> rectangles = retinaFace.infer(imageRgb, 360, 246);

    for (cv::Rect rectangle: rectangles)
    {
        std::cout << rectangle.x << " " << rectangle.y << " " << rectangle.width << " " << rectangle.height << std::endl;
    }

    return 0;
}
