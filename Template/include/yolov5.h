//
// Created by zyh on 12/12/22.
//

#ifndef AI_TEMPLATE_YOLOV5_H
#define AI_TEMPLATE_YOLOV5_H

#define VERSION 7.0

#include <opencv2/opencv.hpp>
#include <random>
#include <utility>

namespace nn {
    template <typename Inference>
    class YoloV5 {
    private:
        YoloV5() noexcept : infer_(nullptr), nms_threshold_(.5f), cc_nms_threshold_(.99f),
                            bs_(0), vector_num_(0), vector_length_(0), num_classes_(0) {}

        Error CheckInput(const std::vector<cv::Mat>& images) noexcept {
            if (bs_ != images.size())    return Error::ErrorInvalidInput;
            for (const auto& img: images)
                if (img.empty() || ((img.channels() != 3) && (img.channels() != 1)) ||
                    img.rows < 20 || img.cols < 20)
                    return Error::ErrorInvalidInput;

            return Success;
        }

        Error CheckAndSetInfer(std::shared_ptr<Inference> infer) noexcept {
            //! Check inputs info.
            const auto& inputs_info = infer->GetInputInfo();
            if (inputs_info.size() != 1)        return Error::ErrorInvalidModelFile;

            const auto images_info = infer->GetInputInfoByName("images");
            if (!images_info.has_value() || images_info->shape.size() != 4)
                return Error::ErrorInvalidModelFile;

            const auto &i_shape = images_info->shape;
            itensor_size_ = {static_cast<int>(i_shape[3]), static_cast<int>(i_shape[2])};

            //! Check outputs info.
            const auto& outputs_info = infer->GetOutputInfo();
            if (outputs_info.size() != 1)       return Error::ErrorInvalidModelFile;

            const auto o_info = infer->GetOutputInfoByName("output0");
            if (!o_info.has_value())            return Error::ErrorInvalidModelFile;

            const auto &o_shape = o_info->shape;
            if (o_shape.size() != 3)            return Error::ErrorInvalidModelFile;

            if (o_shape[0] <= 0 || o_shape[1] <= 0 || o_shape[2] < 6)
                return Error::ErrorInvalidModelFile;

            bs_ = static_cast<size_t>(o_shape[0]);
            vector_num_ = static_cast<size_t>(o_shape[1]);
            vector_length_ = static_cast<size_t>(o_shape[2]);    // [x, y, w, h, obj, cls0, cls1...]
            num_classes_ = vector_length_ - 5;

            infer_ = infer;

            return Error::Success;
        }

        void Letterbox(cv::Mat* img, const cv::Size& target_shape) noexcept {
            const cv::Size shape = {img->cols, img->rows};      // w, h
            const float ratio = std::min(static_cast<float>(target_shape.width) / static_cast<float>(shape.width),
                                         static_cast<float>(target_shape.height) / static_cast<float>(shape.height));    // w, h

            const cv::Size new_unpad = {static_cast<int>(std::round(static_cast<float>(shape.width) * ratio)),
                                        static_cast<int>(std::round(static_cast<float>(shape.height) * ratio))};       // w, h

            if ((shape.width != new_unpad.width) && (shape.height != new_unpad.height))
                cv::resize(*img, *img, new_unpad);

            const cv::Size2f delta = {static_cast<float>(target_shape.width - new_unpad.width) / 2.0f,
                                      static_cast<float>(target_shape.height - new_unpad.height) / 2.0f};

            const int top_bottom[2] = {static_cast<int>(std::round(delta.height - 0.1)),
                                       static_cast<int>(std::round(delta.height + 0.1))};
            const int left_right[2] = {static_cast<int>(std::round(delta.width - 0.1)),
                                       static_cast<int>(std::round(delta.width + 0.1))};

            if (top_bottom[0] || top_bottom[1] || left_right[0] || left_right[1])
                copyMakeBorder(*img, *img, top_bottom[0], top_bottom[1], left_right[0], left_right[1],
                               cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }

        cv::Mat Preprocess(const std::vector<cv::Mat>& src) noexcept {
            std::vector<cv::Mat> new_src;
            for (size_t i = 0; i < src.size(); ++i) {
                new_src.emplace_back(src[i].clone());
                if (new_src[i].channels() == 1)     cv::cvtColor(new_src[i], new_src[i], cv::COLOR_GRAY2BGR);
                Letterbox(&new_src[i], itensor_size_);
            }

            return cv::dnn::blobFromImages(new_src, 1.0f / 255, itensor_size_, cv::Scalar(), true);
        }

        cv::Rect ScaleCoords(const cv::Rect &rect, const cv::Size &src_shape, const cv::Size &target_shape) noexcept {
            const float gain = std::min((float)src_shape.width * 1.0f / (float)target_shape.width,
                                   (float)src_shape.height * 1.0f / (float)target_shape.height);

            const cv::Size2f pad = {((float)src_shape.width - (float)target_shape.width * gain) / 2,
                                    ((float)src_shape.height - (float)target_shape.height * gain) / 2};

            float x1 = (float)rect.x, y1 = (float)rect.y, x2 = x1 + (float)rect.width, y2 = y1 + (float)rect.height;
            const float max_x = (float)(target_shape.width), max_y = (float)(target_shape.height), _min = 0;

            x1 = std::min((x1 - pad.width) / gain, max_x);
            x2 = std::min((x2 - pad.width) / gain, max_x);
            y1 = std::min((y1 - pad.height) / gain, max_y);
            y2 = std::min((y2 - pad.height) / gain, max_y);

            x1 = std::max(x1, _min);
            x2 = std::max(x2, _min);
            y1 = std::max(y1, _min);
            y2 = std::max(y2, _min);

            const int _x1 = (int)(round(x1)), _x2 = (int)(round(x2)),
                    _y1 = (int)(round(y1)), _y2 = (int)(round(y2));

            return {_x1, _y1, _x2 - _x1, _y2 - _y1};

        }

        std::vector<std::vector<NNOutput>> Decode(const float **tensor, const std::vector<cv::Mat>& src) noexcept {
            const float* data = tensor[0];     // YoloV5 has only one output tensor.

            std::vector<std::vector<NNOutput>> boxes(bs_);   // bs, n
            std::vector<size_t> bs_idx(bs_, 0);
            std::iota(bs_idx.begin(), bs_idx.end(), 0);

            // 遍历batch size
            std::for_each(bs_idx.begin(), bs_idx.end(),
                          [data, &boxes, &src, this] (const size_t bs_id) {
                const float* bs_data = data + bs_id * vector_num_ * vector_length_;   // Data pointer of current batch
                std::vector<cv::Rect> rects0;
                std::vector<cv::Rect> offset_rects;
                std::vector<float> confs0;
                std::vector<std::int64_t> ids0;

                std::vector<size_t> vector_idx(vector_num_, 0);
                std::iota(vector_idx.begin(), vector_idx.end(), 0);

                // 遍历vector num
                std::for_each(vector_idx.begin(), vector_idx.end(), [bs_data, &rects0, &offset_rects, &confs0, &ids0, this] (const size_t& vector_id) {
                    const float* vec_data = bs_data + vector_id * vector_length_;
                    float obj_conf = vec_data[4];
                    const float* max_conf = std::max_element(vec_data + 5, vec_data + vector_length_);
                    float conf = *max_conf * obj_conf;
                    std::int64_t class_id = max_conf - (vec_data + 5);
                    if (conf < cls_thresholds_[class_id])       return;

                    confs0.emplace_back(conf);
                    ids0.emplace_back(class_id);

                    float center_x = *vec_data;
                    float center_y = vec_data[1];
                    float box_w = vec_data[2];
                    float box_h = vec_data[3];
                    int x0 = static_cast<int>(center_x - box_w / 2);
                    int y0 = static_cast<int>(center_y - box_h / 2);
                    int x0_offset = x0 + 4096 * static_cast<int>(class_id);
                    int y0_offset = y0 + 4096 * static_cast<int>(class_id);

                    rects0.emplace_back(x0, y0, static_cast<int>(box_w), static_cast<int>(box_h));
                    offset_rects.emplace_back(x0_offset, y0_offset,
                                              static_cast<int>(box_w), static_cast<int>(box_h));
                });
                std::vector<int> indices;
                cv::dnn::NMSBoxes(offset_rects, confs0, 0, nms_threshold_, indices);

                std::vector<cv::Rect> rects1;
                std::vector<float> confs1;
                std::vector<std::int64_t> ids1;
                for (const int ix : indices) {
                    rects1.emplace_back(rects0[ix]);
                    confs1.emplace_back(confs0[ix]);
                    ids1.emplace_back(ids0[ix]);
                }

                indices.clear();
                cv::dnn::NMSBoxes(rects1, confs1, 0, cc_nms_threshold_, indices);

                std::vector<NNOutput>& box = boxes[bs_id];      // Container to store current batch result
                for (const int ix : indices)
                    box.emplace_back(confs1[ix], ids1[ix],
                                     ScaleCoords(rects1[ix], itensor_size_, src[bs_id].size()), class_names_[ids1[ix]]);
            });

            return boxes;
        }

        // 手动设置
        float nms_threshold_;
        float cc_nms_threshold_;
        std::vector<std::string> class_names_;
        std::vector<float> cls_thresholds_;

        // 加载infer时设置
        size_t bs_;
        size_t vector_num_;
        size_t vector_length_;
        size_t num_classes_;
        cv::Size itensor_size_;

        std::shared_ptr<Inference> infer_;
    public:
        ~YoloV5() noexcept = default;

        YoloV5(YoloV5&&) noexcept = default;

        YoloV5& operator=(YoloV5&&) noexcept = default;

        YoloV5(const YoloV5&) noexcept = delete;

        YoloV5& operator=(const YoloV5&) noexcept = delete;

        Error SetClsThreshold(float cls_threshold) noexcept {
            if (cls_threshold < .0f || cls_threshold > 1.0f)    return Error::ErrorInvalidNNConfig;

            cls_thresholds_.assign(num_classes_, cls_threshold);

            return Error::Success;
        }

        Error SetClsThreshold(std::vector<float> cls_thresholds) noexcept {
            for (const float conf : cls_thresholds)
                if (conf < .0f || conf > 1.0f)          return Error::ErrorInvalidNNConfig;
            if (cls_thresholds.size() != num_classes_)  return Error::ErrorInvalidNNConfig;
            cls_thresholds_ = std::move(cls_thresholds);

            return Error::Success;
        }

        Error SetClassNames(std::vector<std::string> names) noexcept {
            if (names.size() != num_classes_)  return Error::ErrorInvalidNNConfig;
            class_names_ = std::move(names);

            return Error::Success;
        }

        Error SetNmsThreshold(float nms_threshold) noexcept {
            if (nms_threshold < .0f || nms_threshold > 1.0f)
                return Error::ErrorInvalidNNConfig;
            nms_threshold_ = nms_threshold;

            return Error::Success;
        }

        Error SetCCNmsThreshold(float cc_nms_threshold) noexcept {
            if (cc_nms_threshold < .0f || cc_nms_threshold > 1.0f)
                return Error::ErrorInvalidNNConfig;
            cc_nms_threshold_ = cc_nms_threshold;

            return Error::Success;
        }

        std::vector<std::vector<NNOutput>> Exec(const std::vector<cv::Mat> &i_tensors, Error *ec) noexcept {
            Error err = CheckInput(i_tensors);
            if (err) {
                *ec = err;
                return {};
            }

            if (cls_thresholds_.empty() || class_names_.empty()) {
                *ec = Error::ErrorInvalidNNConfig;
                return {};
            }

            const std::vector<std::string> input_names = {"images"};
            const std::vector<std::string> output_names = {"output0"};

            cv::Mat tensor = Preprocess(i_tensors);

            std::cout << "\nInput: " << std::endl;
            std::vector<cv::Range> range0({cv::Range::all(), cv::Range(0, 1), cv::Range::all(), cv::Range::all()});
            std::cout << "Input ch0 sum: " << std::setprecision(10) << cv::sum(tensor(range0))[0] << std::endl;

            std::vector<cv::Range> range1({cv::Range::all(), cv::Range(1, 2), cv::Range::all(), cv::Range::all()});
            std::cout << "Input ch1 sum: " << std::setprecision(10) << cv::sum(tensor(range1))[0] << std::endl;

            std::vector<cv::Range> range2({cv::Range::all(), cv::Range(2, 3), cv::Range::all(), cv::Range::all()});
            std::cout << "Input ch2 sum: " << std::setprecision(10) << cv::sum(tensor(range2))[0] << std::endl;

            const float *iptr = (float*)tensor.data;

            std::vector<float> outputs(bs_ * vector_num_ * vector_length_);
            float* optr = outputs.data();

            err = infer_->Run(&iptr, input_names, &optr, output_names);
            if (err) {
                *ec = err;
                return {};
            }

            std::cout << "\nOuts: " << std::endl;
            float out_sum = .0f;
            for (auto t: outputs)   out_sum += t;
            std::cout << "out: " << std::setprecision(15) << out_sum << std::endl;

            return Decode(const_cast<const float**>(&optr), i_tensors);
        }

        Error Visualize(cv::Mat *img, const std::vector<NNOutput> &outputs) {
            if (img->channels() == 1)
                cv::cvtColor(*img, *img, cv::COLOR_GRAY2BGR);

            Error err = Error::Success;
            std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
            std::int64_t seek = std::chrono::system_clock::to_time_t(tp);
            static std::default_random_engine random_eng(seek % 100);
            std::uniform_int_distribution<std::uint8_t> ud(0, 255);

            cv::Mat &draw_img = *img;
            const int font_face = cv::FONT_HERSHEY_COMPLEX, font_thickness = 1;
            int _baseline = 0;
            const double font_scale = 0.5;

            const int rect_thickness = 2;

            std::vector<cv::Scalar> colors;
            for (size_t i = 0; i < outputs.size(); ++i) {
                colors.emplace_back(ud(random_eng), ud(random_eng), ud(random_eng));

                const NNOutput &output = outputs[i];
                const auto& bbox = output.bbox;
                //! Draw the bounding box on the image.
                cv::rectangle(draw_img, bbox, colors[i], rect_thickness);

                std::stringstream ss;
                ss << output.class_name << " " << std::setprecision(2) << output.conf;
                auto text_size = cv::getTextSize(ss.str(), font_face, font_scale, font_thickness, &_baseline);
                text_size.width += font_thickness;
                text_size.height += font_thickness;

                if (text_size.width > draw_img.cols || text_size.height > draw_img.rows) {
                    err = Error::ErrorVisualizeError;
                    continue;
                }
                int text_x = bbox.x - rect_thickness / 2;
                if (text_x < 0)         text_x = bbox.x;
                if (text_x + text_size.width >= draw_img.cols)
                    text_x = bbox.x + bbox.width - text_size.width;
                if (text_x < 0)         text_x = 0;

                int text_y = bbox.y - text_size.height;     // 左上角外面
                if (text_y < 0)             text_y = bbox.y + bbox.height;  // 左下角外面
                if (text_y + text_size.height >= draw_img.rows)     text_y = bbox.y;    // 左上角里面

                //! Draw the text and background on the image.
                draw_img(cv::Rect(cv::Point(text_x, text_y), text_size)).setTo(colors[i]);
                text_y += text_size.height - font_thickness * 2;
                cv::putText(draw_img, ss.str(), cv::Point(text_x, text_y), font_face, font_scale,
                            cv::Scalar(255, 255, 255), font_thickness);
            }

            return err;
        }

        std::string Print() noexcept {
            return infer_->Print();
        }

        static std::shared_ptr<YoloV5> CreateModel(Error* ec, std::shared_ptr<Inference> infer) noexcept {
            using YoloV5_ = YoloV5<Inference>;
            std::shared_ptr<YoloV5_> yolov5_(new YoloV5_());

            Error err = yolov5_->CheckAndSetInfer(infer);
            if (err) {
                *ec = err;

                return nullptr;
            }

            return yolov5_;
        }
    };
}

#endif //AI_TEMPLATE_YOLOV5_H
