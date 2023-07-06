#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class YoloDetector {
public:
    YoloDetector(const std::string& model_cfg, const std::string& model_weights, const std::vector<std::string>& classes);
    cv::Rect get_output_format(const cv::Rect& box);
    std::map<std::string, std::vector<cv::Rect>> detect(const cv::Mat& img, float conf = 0.2, float nms_thresh = 0.3, bool non_max_suppression = true, const std::vector<float>& class_conf = {});
private:
    cv::dnn::Net net;
    std::vector<std::string> classes;
    std::vector<std::string> layer_names;
    std::vector<std::string> outputlayers;
};

YoloDetector::YoloDetector(const std::string& model_cfg, const std::string& model_weights, const std::vector<std::string>& classes) : classes(classes) {
    net = cv::dnn::readNetFromDarknet(model_cfg, model_weights);
    int count = cv::cuda::getCudaEnabledDeviceCount();
    if (count > 0) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "GPU Backend selected" << std::endl;
    }
    layer_names = net.getLayerNames();
    outputlayers = { layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers() };
}

cv::Rect YoloDetector::get_output_format(const cv::Rect& box) {
    int x = box.x;
    int y = box.y;
    int w = box.width;
    int h = box.height;
    return cv::Rect(x, y, w, h);
}

std::map<std::string, std::vector<cv::Rect>> YoloDetector::detect(const cv::Mat& img, float conf, float nms_thresh, bool non_max_suppression, const std::vector<float>& class_conf) {
    std::map<std::string, std::vector<cv::Rect>> final_result;
    std::map<std::string, std::vector<float>> confidences;
    std::map<std::string, std::vector<cv::Rect>> boxes;
    cv::Mat blob = cv::dnn::blobFromImage(img, 0.00392, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, outputlayers);
    int Height = img.rows;
    int Width = img.cols;
    for (const auto& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            float* data = (float*)out.ptr<float>(i);
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            int class_id = class_id_point.x;
            confidence = scores.at<float>(0, class_id);
            if (confidence > conf) {
                cv::Rect box;
                box.x = static_cast<int>(data[0] * Width);
                box.y = static_cast<int>(data[1] * Height);
                box.w = static_cast<int>(data[2] * Width);
                box.h = static_cast<int>(data[3] * Height);
                confidences[classes[class_id]].push_back(static_cast<float>(confidence));
                boxes[classes[class_id]].push_back(box);
            }
        }
    }
    std::map<std::string, std::vector<int>> indices;
    if (non_max_suppression) {
        for (const auto& pair : boxes) {
            const std::string& class_name = pair.first;
            const std::vector<cv::Rect>& box = pair.second;
            std::vector<float>& conf = confidences[class_name];
            float class_conf_val = (class_conf.size() > 0) ? class_conf[class_id] : conf;
            std::vector<int> selected_indices;
            cv::dnn::NMSBoxes(box, conf, class_conf_val, nms_thresh, selected_indices);
            indices[class_name] = selected_indices;
        }
    }
    else {
        for (const auto& pair : boxes) {
            const std::string& class_name = pair.first;
            const std::vector<cv::Rect>& box = pair.second;
            std::vector<int> all_indices(box.size());
            std::iota(all_indices.begin(), all_indices.end(), 0);
            indices[class_name] = all_indices;
        }
    }
    for (const auto& pair : indices) {
        const std::string& key = pair.first;
        const std::vector<int>& index = pair.second;
        for (int i : index) {
            int select = i;
            final_result[key].push_back(get_output_format(boxes[key][select]));
        }
    }
    return final_result;
}

int main() {
    // Read the default classes for the YOLO model
    std::ifstream file("./classes.names");
    std::vector<std::string> classes;
    std::string line;
    while (std::getline(file, line)) {
        classes.push_back(line);
    }
    std::cout << "Default classes: " << std::endl;
    for (int n = 0; n < classes.size(); ++n) {
        std::cout << n + 1 << ". " << classes[n] << std::endl;
    }

    // Select specific classes that you want to detect out of the 80 and assign a color to each detection
    std::map<std::string, cv::Scalar> selected;
    selected["bunkdgdfgdfer"] = cv::Scalar(0, 255, 255);
    selected["furncbcbdace_tilt"] = cv::Scalar(0, 0, 0);
    selected["ladfgdfdle"] = cv::Scalar(255, 0, 0);

    // Initialize the detector with the paths to cfg, weights, and the list of classes
    YoloDetector detector("./fsdfssdface_yolo.cfg", "./sdfsdfs.weights", classes);

    // Initialize video stream
    cv::VideoCapture cap("./dgdfg.mp4");

    // Read the first frame
    cv::Mat frame;
    cap.read(frame);

    // Loop to read frames and update window
    while (!frame.empty()) {
        double start = cv::getTickCount();

        // This returns detections in the format {cls_1:[(top_left_x, top_left_y, top_right_x, top_right_y), ..],
        //                                        cls_4:[], ..}
        // Note: you can change the file as per your requirement if necessary
        std::map<std::string, std::vector<cv::Rect>> detections = detector.detect(frame);

        //Loop over the selected items and check if it exists in the detected items,
        // if it exists loop over all the items of the specific class and draw rectangles and put a label in the defined color
        for (const auto& pair : selected) {
            const std::string& cls = pair.first;
            const cv::Scalar& color = pair.second;
            if (detections.find(cls) != detections.end()) {
                const std::vector<cv::Rect>& boxes = detections[cls];
                for (const cv::Rect& box : boxes) {
                    cv::rectangle(frame, box, color, 1);
                    cv::putText(frame, cls, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
                }
            }
        }

        double end = cv::getTickCount();
        double fps = cv::getTickFrequency() / (end - start);
        cv::putText(frame, "fps: " + std::to_string(fps), cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(100, 0, 0));

        // Display the detections
        cv::imshow("detections", frame);

        // Wait for key press
        int key_press = cv::waitKey(1) & 0xff;

        // Exit loop if 'q' is pressed or on reaching EOF
        if (key_press == 'q') {
            break;
        }

        cap.read(frame);
    }

    // Release resources
    cap.release();

    // Destroy window
    cv::destroyAllWindows();

    return 0;
}

