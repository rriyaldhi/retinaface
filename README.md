# RetinaFace

The implementation is based on https://github.com/wang-xinyu/Pytorch_Retinaface and https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface.

## Dependencies

Below is the installation script for the required dependencies on Ubuntu Server 18.04.

### Cuda 11.1
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
rm cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
```

### TensorRT
Download cuda 11.1 from: https://drive.google.com/uc?id=1fvoEa3BfAOvlDTsi9MByHtj5ciRjQcQY
```
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y tensorrt
rm nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
```

### OpenCV

```
sudo apt-get install -y build-essential
sudo apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y python3-dev python3-numpy python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt update
sudo apt install -y libjasper-dev
wget https://github.com/opencv/opencv/archive/4.3.0.zip
unzip 4.3.0.zip
rm 4.3.0.zip
mv opencv-4.3.0 opencv
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
sudo make install
```

## Example Usage
Example usage can be found in [inference.cpp](https://github.com/rriyaldhi/retinaface/blob/main/inference.cpp).

## Build & Execute

Follow this [guide](https://github.com/rriyaldhi/retinaface/blob/main/engine/README.md) to generate retinaface.engine

```
mkdir build && cd build
cmake ..
make
cp <REPOSITORY_ROOT>/engine/build/retinaface.engine .
./inference
```
