#include <iostream>
#include <vector>

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Argument is required!" << std::endl;
        std::cerr << "./inference /path/to/image" << std::endl;
        return -1;
    }

    RetinaFace retinaFace = RetinaFace();
    std::vector<cv::Rect> rectangles = retinaFace.infer(std::string(argv[1]));

    for (cv::Rect rectangle: rectangles)
    {
        std::cout << rectangle.x << " " << rectangle.y << " " << rectangle.width << " " << rectangle.height << std::endl;
    }

    return 0;
}
