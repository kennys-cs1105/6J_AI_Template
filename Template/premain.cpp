//
// Created by zyh on 11/15/22.
//

#include <thread>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <string_view>

#include "include/inference.h"
#include "include/yolov5.h"
#include "include/yolov5_seg.h"
#include "include/yolact.h"

int main() {
    //! Pascal classes.
    std::initializer_list<std::string> pascal_classes({"aeroplane", "bicycle", "bird", "boat", "bottle",
                                                                "bus", "car", "cat", "chair", "cow", "diningtable",
                                                                "dog", "horse", "motorbike", "person", "pottedplant",
                                                                "sheep", "sofa", "train", "tvmonitor"});
    //! Coco classes.
    std::initializer_list<std::string> coco_classes({"person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                                     "train", "truck", "boat", "traffic light", "fire hydrant",
                                                     "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                                     "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                                     "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                                     "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                                     "baseball glove", "skateboard", "surfboard", "tennis racket",
                                                     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                                     "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                                     "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                                     "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                                     "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                                     "scissors", "teddy bear", "hair drier", "toothbrush"});

    //! Create model.
//    nn::Error err;
//    nn::OnnxRT::Config onnx_config("../yolov5/yolov5x-seg.onnx", "yolov5");
//    std::shared_ptr<nn::OnnxRT> onnx = nn::OnnxRT::MakeEngine(&err, onnx_config);
//    auto net = nn::YoloV5_Seg<nn::OnnxRT>::CreateModel(&err, onnx);

    nn::Error err;
    nn::TensorRT::Config config("../yolov5/yolov5m.trtengine", "yolov5");
    config.SetCudaDeviceConcurrency(0, 1);
    std::shared_ptr<nn::TensorRT> engine = nn::TensorRT::MakeEngine(&err, config);
    auto net = nn::YoloV5<nn::TensorRT>::CreateModel(&err, engine);

    if (err) {
        std::cerr << "Create error: " << err << std::endl;
        std::terminate();
    }

    std::cout << net->Print() << std::endl << std::endl;

    net->SetClassNames(coco_classes);
//    net->SetClassNames({"A", "B", "C"});
    net->SetCCNmsThreshold(0.99);
    net->SetClsThreshold(0.25);
    net->SetNmsThreshold(0.45);


    //! Read the image from the disk and inference.
    std::filesystem::path path("../");
    std::filesystem::directory_iterator iter(path);
    for (const std::filesystem::directory_entry& it : iter) {
        const std::string img_path = it.path().string();
	std::cout << img_path << std::endl;
        if (img_path.size() <= 4 || img_path.substr(img_path.size() - 4, 4) != ".jpg") {
            std::cerr << "The file does not ends with .jpg" << std::endl;
            // return 1;
	    continue;
        }

        const std::string img_filename(it.path().filename());
//        if (img_filename != "000000519764.jpg")     continue;

        cv::Mat img = cv::imread(img_path);
        cv::Mat draw = img.clone();
        std::cout << img_filename << " " << img.size() << std::endl;

        std::vector<cv::Mat> inputs;
        for (size_t t = 0; t < 1; ++t)
            inputs.emplace_back(img.clone());

        auto time0 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<nn::NNOutput>> res = net->Exec(inputs, &err);
        if (err) {
            std::cerr << "Exec error: " << err << std::endl;
            std::terminate();
        }

        auto time1 = std::chrono::high_resolution_clock::now();
        auto delta = time1 - time0;

        std::cout << "Decoded outputs: " << std::endl;
        for (auto& r : res[0])
            std::cout << r << std::endl;

        std::cout << "Cost: " <<
                  static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()) / 1000.0f << std::endl;

        err = net->Visualize(&draw, res[0]);
        if (err) {
            std::cerr << "Visualize error: " << err << std::endl;
//            std::terminate();
        }

        std::filesystem::path save_path("../saved");
        save_path.append(img_filename);

        cv::imwrite(save_path, draw);
//
//        std::cout << std::endl;
//    }
//
////    std::this_thread::sleep_for(std::chrono::seconds(10));
//
//    return 0;
//}
//



