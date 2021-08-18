#include <iostream>
#include <vector>
#include "retinaface.h"

int main(int argc, char** argv) {
    RetinaFace retinaFace = RetinaFace();
    std::vector<cv::Rect> rectangles = retinaFace.inferVideo(std::string(argv[1]), std::string(argv[2]));
    return 0;
}
