//
// Created by zyh on 12/9/22.
//

#ifndef AI_TEMPLATE_BASE_H
#define AI_TEMPLATE_BASE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace nn {
    class NNOutput {
    public:
        friend std::ostream &operator<<(std::ostream &output, const NNOutput &out) {
            output << out.class_name;
            output << " " << std::setprecision(5) << out.conf;

            if (!out.bbox.empty()) {
                output << ": " << out.bbox.x;
                output << " " << out.bbox.y;
                output << " " << out.bbox.x + out.bbox.width;
                output << " " << out.bbox.y + out.bbox.height;
            }

            if (!out.mask.empty())  output << ", mask sum: " << std::setprecision(10) << cv::sum(out.mask)[0];

            return output;
        }

        NNOutput(float _conf, int _class_id, cv::Rect _bbox, cv::Mat _mask, std::string _class_name) :
                conf(_conf), class_id(_class_id), bbox(std::move(_bbox)),
                mask(std::move(_mask)), class_name(std::move(_class_name)) {}
        NNOutput(float _conf, int _class_id, cv::Rect _bbox, std::string _class_name) :
                conf(_conf), class_id(_class_id), bbox(std::move(_bbox)), class_name(std::move(_class_name)) {}

        float conf;
        int class_id;
        cv::Rect bbox;
        cv::Mat mask;
        std::string class_name;
    };

    template<typename T>
    class TensorInfo_ {
    public:
        TensorInfo_(std::string name, std::vector<T> shape) :
                name(std::move(name)), shape(std::move(shape)) {}

        static_assert(std::is_integral<T>());

        const std::string name;
        const std::vector<T> shape;
    };

    enum Error {
        Success = 0,
        // 1-9: Inference engine error
        ErrorInvalidModelFile = 1,
        ErrorInferenceError = 2,
        ErrorInferEnvError = 3,
        ErrorInvalidTensor = 4,

        // 10-19: Neural network error
        ErrorInvalidInput = 10,
        ErrorInvalidNNConfig = 11,
        ErrorVisualizeError = 12,
    };
}

#endif //AI_TEMPLATE_BASE_H











