#include <iostream>
#include <vector>
#include "retinaface.h"

int main(int argc, char** argv) {
    RetinaFace retinaFace = RetinaFace();
    retinaFace.inferVideo(std::string(argv[1]), std::string(argv[2]));
    return 0;
}
