
# Yolov5 with openCV DNN in C++

Try out openCV DNN API with Yolov5 model



## Requirement
    1. Ubuntu 22.04
    2. CUDA 11.8
    3. cuDNN 8.7.0
    4. g++ 11.3
    5. cmake 3.22.1
    6. OpenCV 4.6.0 (with CUDA backend - optional)




## Usage/Examples


#### Step1 

Comment these lines (in ***main.cpp***) if you do not build openCV with CUDA backend
```cpp
cout << "Attempt to use CUDA\n";
net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
```


#### Step 2

In your command line, type
```bash
.../<project_dir>$ cmake .
.../<project_dir>$ make
```


#### Step 3

To run the script, type in your command line
```bash
.../<project_dir>$ ./yolov5_cpp
```
