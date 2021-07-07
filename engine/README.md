# RetinaFace - Generate Engine

1. Download retinaface.wts and put in <REPOSITORY_ROOT>/engine directory: https://drive.google.com/uc?id=17VxDFbPL2kc-ChUG6KzKBNUVzTA0avWF

2. Build the engine generator source
```
$ mkdir build && cd build
$ cmake ..
$ make
```

3. Download images for calibration and put in <REPOSITORY_ROOT>/engine/build directory: https://drive.google.com/uc?id=1YardM13gsB7XeSirXEb3ApbXhbSLJmIB
```
$ unzip widerface_calib.zip
```

4. Generate the engine
```
$ ./generate
```
