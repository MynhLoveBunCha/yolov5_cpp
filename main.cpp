#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;


// set global params
const vector<Scalar> colors = {Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0)};
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;


struct Detection
{
    int class_id;
    float confidence;
    Rect box;
};


vector<string> load_class_list()
{
    vector<string> class_list;
    ifstream ifs("classes.txt");
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}


Mat format_yolov5(const Mat &source){
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}


void detect(Mat &image, dnn::Net &net, vector<Detection> &output, const vector<string> &className) {
    Mat blob;
    auto input_image = format_yolov5(image);
    dnn::blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);
    std::vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            Mat scores(1, className.size(), CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}


int main(){
    // init webcam capture //
    Mat frame;
    VideoCapture cap;
    int camID = 2;  // your cam ID might be 0, 1 or 2
    int defaultID = CAP_ANY;
    cap.open(camID, defaultID);
    if(!cap.isOpened()){
        cout << "No camera detected!" << endl;
        return -1;
    }

    // load model //
    auto net = dnn::readNet("yolov5s.onnx");

    // Comment the following 3 lines if you do not build opencv with CUDA backend
    // ---------------------------
    cout << "Attempt to use CUDA\n";
    net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
    // ---------------------------


    // Get list of class name
    vector<string> class_list = load_class_list();


    // init loop
    while(1){
        bool isFrame = cap.read(frame);
        if(!isFrame){
            cout << "No frame!" << endl;
            break;
        }
        auto start = chrono::high_resolution_clock::now();


        // Inference 
        vector<Detection> output;
        detect(frame, net, output, class_list);
        int n_detections = output.size();


        // Draw bounding box
        for (int i = 0; i < n_detections; ++i)
        {

            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            auto confi = detection.confidence;
            const auto color = colors[classId % colors.size()];
            rectangle(frame, box, color, 3);

            rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), color, FILLED);
            putText(frame, class_list[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            putText(frame, to_string(confi * 100.) + "%", Point(box.x + 70, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        }
        auto end = chrono::high_resolution_clock::now();


        // FPS calculation
        float fps = 1 * 1000.0 / chrono::duration_cast<chrono::milliseconds>(end - start).count();
        ostringstream fps_label;
        fps_label << fixed << setprecision(2);
        fps_label << "FPS: " << fps;
        string fps_label_str = fps_label.str();

        cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);


        // display
        imshow("webcam", frame);


        // Exit condition
        if(waitKey(15) >= 0){
            break;
        }
    }
    return 0;
}