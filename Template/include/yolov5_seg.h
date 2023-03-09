//
// Created by zyh on 12/12/22.
//

#ifndef AI_TEMPLATE_YOLOV5_SEG_H
#define AI_TEMPLATE_YOLOV5_SEG_H

#define VERSION 7.0

#include <opencv2/opencv.hpp>
#include <random>
#include <utility>

namespace nn {
    template <typename Inference>
    class YoloV5_Seg {
    private:
        YoloV5_Seg() noexcept : infer_(nullptr), nms_threshold_(.5f), cc_nms_threshold_(.99f),
                            bs_(0), vector_num_(0), vector_length_(0), mask_dim_(0),
                            mask_h_(0), mask_w_(0), num_classes_(0) {}

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

            //! itensor_size_ is the static dimension, it will be upgraded to dynamic in the update.
            const auto &i_shape = images_info->shape;
            itensor_size_ = {static_cast<int>(i_shape[3]), static_cast<int>(i_shape[2])};

            //! Check outs info.
            const auto& outputs_info = infer->GetOutputInfo();
            if (outputs_info.size() != 2)       return Error::ErrorInvalidModelFile;

            //! Check proto info.
            const auto& proto_info = infer->GetOutputInfoByName("output1");
            if (!proto_info.has_value())            return Error::ErrorInvalidModelFile;

            const auto& proto_shape = proto_info->shape;
            if (proto_shape.size() != 4)            return Error::ErrorInvalidModelFile;

            if (proto_shape[0] <= 0 || proto_shape[1] <= 0 || proto_shape[2] <= 0 ||
                proto_shape[3] <= 0 || proto_shape[2] != proto_shape[3])
                return Error::ErrorInvalidModelFile;

            mask_dim_ = static_cast<size_t>(proto_shape[1]);

            //! proto_h_ and proto_w_ is the static dimension, it will be upgraded to dynamic in the update,
            mask_h_ = static_cast<size_t>(proto_shape[2]);
            mask_w_ = static_cast<size_t>(proto_shape[3]);

            //! Check pred info.
            const auto vector_info = infer->GetOutputInfoByName("output0");
            if (!vector_info.has_value())            return Error::ErrorInvalidModelFile;

            const auto &vector_shape = vector_info->shape;
            if (vector_shape.size() != 3)            return Error::ErrorInvalidModelFile;

            if (vector_shape[0] <= 0 || vector_shape[1] <= 0 || vector_shape[2] < 6)
                return Error::ErrorInvalidModelFile;

            bs_ = static_cast<size_t>(vector_shape[0]);
            vector_num_ = static_cast<size_t>(vector_shape[1]);
            vector_length_ = static_cast<size_t>(vector_shape[2]);
            num_classes_ = vector_length_ - mask_dim_ - 5;

            infer_ = infer;

            return Error::Success;
        }

        void Letterbox(cv::Mat* img, const cv::Size2f& target_shape) noexcept {
            const cv::Size2f shape = img->size();      // w, h
            const float ratio = std::min(target_shape.width / shape.width,
                                         target_shape.height / shape.height);    // w, h

            const cv::Size2f new_unpad = {std::round(shape.width * ratio),
                                        std::round(shape.height * ratio)};       // w, h

            if ((static_cast<int>(shape.width) != static_cast<int>(new_unpad.width)) &&
                (static_cast<int>(shape.height) != static_cast<int>(new_unpad.height)))
                cv::resize(*img, *img, new_unpad);

            const cv::Size2f delta = {(target_shape.width - new_unpad.width) / 2.0f,
                                      (target_shape.height - new_unpad.height) / 2.0f};

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

        cv::Rect ScaleCoords(const cv::Rect2f &rect, const cv::Size2f &src_shape, const cv::Size2f &target_shape) noexcept {
            const float gain = std::min(src_shape.width / target_shape.width,
                                        src_shape.height / target_shape.height);

            const cv::Size2f pad = {(src_shape.width - target_shape.width * gain) / 2.0f,
                                    (src_shape.height - target_shape.height * gain) / 2.0f};

            float x1 = rect.x, y1 = rect.y, x2 = x1 + rect.width, y2 = y1 + rect.height;
            const float max_x = target_shape.width - 1.0f, max_y = target_shape.height - 1.0f, _min = .0f;

            x1 = std::min((x1 - pad.width) / gain, max_x);
            x2 = std::min((x2 - pad.width) / gain, max_x);
            y1 = std::min((y1 - pad.height) / gain, max_y);
            y2 = std::min((y2 - pad.height) / gain, max_y);

            x1 = std::max(x1, _min);
            x2 = std::max(x2, _min);
            y1 = std::max(y1, _min);
            y2 = std::max(y2, _min);

            const int _x1 = static_cast<int>(round(x1)), _x2 = static_cast<int>(round(x2)),
                    _y1 = static_cast<int>(round(y1)), _y2 = static_cast<int>(round(y2));

            return {_x1, _y1, _x2 - _x1, _y2 - _y1};

        }

        //! Rescale from src_shape to target_shape for a mask or an image plotted with a mask.
        void ScaleImage(cv::Mat* img, const cv::Size& src_shape, const cv::Size& target_shape) noexcept {
            const float gain = std::min(static_cast<float>(src_shape.width) / static_cast<float>(target_shape.width),
                                        static_cast<float>(src_shape.height) / static_cast<float>(target_shape.height));

            const cv::Size2f pad = {(static_cast<float>(src_shape.width) - static_cast<float>(target_shape.width) * gain) / 2.0f,
                                    (static_cast<float>(src_shape.height) - static_cast<float>(target_shape.height) * gain) / 2.0f};

            const int top = static_cast<int>(pad.height), left = static_cast<int>(pad.width);
            const int bottom = static_cast<int>(static_cast<float>(src_shape.height) - pad.height);
            const int right = static_cast<int>(static_cast<float>(src_shape.width) - pad.width);

            cv::Mat& img_ = *img;
            cv::Mat roi = img_(cv::Rect(left, top, right - left, bottom - top));
            cv::resize(roi, img_, target_shape);
        }

        std::vector<std::vector<NNOutput>> Decode(const float **tensor, const std::vector<cv::Mat>& src) noexcept {
            std::vector<std::vector<NNOutput>> boxes(bs_);   // bs, n
            std::vector<size_t> bs_idx(bs_, 0);
            std::iota(bs_idx.begin(), bs_idx.end(), 0);

            //! Loop by batch size.
            std::for_each(bs_idx.begin(), bs_idx.end(),
                          [tensor, &boxes, &src, this] (const size_t bs_id) {
                              const auto& img_size = src[bs_id].size();
                              const float* vector_data = tensor[0] + bs_id * vector_num_ * vector_length_;

                              std::vector<cv::Rect2f> rects0_f;
                              std::vector<cv::Rect> rects0;
                              std::vector<cv::Rect> offset_rects;
                              std::vector<float> confs0;
                              std::vector<std::int64_t> ids0;
                              std::vector<std::vector<float>> mask0;
                              std::vector<void*> res_ptrs({&rects0_f, &rects0, &offset_rects, &confs0, &ids0, &mask0});

                              std::vector<size_t> vector_idx(vector_num_, 0);
                              std::iota(vector_idx.begin(), vector_idx.end(), 0);

                              //! Loop by vector num.
                              std::for_each(vector_idx.begin(), vector_idx.end(),
                                            [&res_ptrs, vector_data, this] (const size_t& vector_id) {
                                                const float* vec_data = vector_data + vector_id * vector_length_;
                                                float obj_conf = vec_data[4];
                                                const float* max_conf = std::max_element(vec_data + 5, vec_data + 5 + num_classes_);
                                                float conf = *max_conf * obj_conf;
                                                std::int64_t class_id = max_conf - (vec_data + 5);
                                                if (conf < cls_thresholds_[class_id])       return;

                                                static_cast<std::vector<float>*>(res_ptrs[3])->emplace_back(conf);
                                                static_cast<std::vector<std::int64_t>*>(res_ptrs[4])->emplace_back(class_id);

                                                //! The first four of vec_data are center_x, center_y, box_w and box_h.
                                                //! These are used to generate x, y, w, and h for bounding box.
                                                float xywh_f[] = {*vec_data - vec_data[2] / 2.0f,
                                                                  vec_data[1] - vec_data[3] / 2.0f,
                                                                  vec_data[2], vec_data[3]};
                                                int xywh[] = {static_cast<int>(xywh_f[0]),
                                                              static_cast<int>(xywh_f[1]),
                                                              static_cast<int>(xywh_f[2]),
                                                              static_cast<int>(xywh_f[3])};
                                                int xy_offset[] = {xywh[0] + 4096 * static_cast<int>(class_id),
                                                                   xywh[1] + 4096 * static_cast<int>(class_id)};

                                                static_cast<std::vector<cv::Rect2f>*>(res_ptrs[0])->emplace_back(
                                                        xywh_f[0], xywh_f[1], xywh_f[2], xywh_f[3]);
                                                static_cast<std::vector<cv::Rect>*>(res_ptrs[1])->emplace_back(
                                                        xywh[0], xywh[1], xywh[2], xywh[3]);
                                                static_cast<std::vector<cv::Rect>*>(res_ptrs[2])->emplace_back(
                                                        xy_offset[0], xy_offset[1], xywh[2], xywh[3]);

                                                const float* mask_begin = vec_data + 5 + num_classes_;
                                                const float* mask_end = vec_data + vector_length_;
                                                static_cast<std::vector<std::vector<float>>*>(res_ptrs[5])->emplace_back(mask_begin, mask_end);
                                            });
                              if (offset_rects.empty())       return;

                              std::vector<int> indices;
                              cv::dnn::NMSBoxes(offset_rects, confs0, 0, nms_threshold_, indices);

                              std::vector<cv::Rect2f> rects1_f;
                              std::vector<cv::Rect> rects1;
                              std::vector<float> confs1;
                              std::vector<std::int64_t> ids1;
                              std::vector<std::vector<float>> mask1;
                              for (const int ix : indices) {
                                  rects1_f.emplace_back(rects0_f[ix]);
                                  rects1.emplace_back(rects0[ix]);
                                  confs1.emplace_back(confs0[ix]);
                                  ids1.emplace_back(ids0[ix]);
                                  mask1.emplace_back(std::move(mask0[ix]));
                              }

                              indices.clear();
                              cv::dnn::NMSBoxes(rects1, confs1, 0, cc_nms_threshold_, indices);

                              std::vector<cv::Rect2f> rects2_f;
                              std::vector<float> confs2;
                              std::vector<std::int64_t> ids2;
                              std::vector<float> mask_fuse0;
                              for (const int ix : indices) {
                                  rects2_f.emplace_back(rects1_f[ix]);
                                  confs2.emplace_back(confs1[ix]);
                                  ids2.emplace_back(ids1[ix]);
                                  mask_fuse0.insert(mask_fuse0.end(), mask1[ix].begin(), mask1[ix].end());
                              }

                              const float* proto_data = tensor[1] + bs_id * mask_dim_ * mask_h_ * mask_w_;
                              const int mask_dim = static_cast<int>(mask_dim_);
                              const int mask_h = static_cast<int>(mask_h_);
                              const int mask_w = static_cast<int>(mask_w_);
                              const int n = static_cast<int>(rects2_f.size());

                              const std::vector<int> proto_shape({mask_dim, mask_h * mask_w});
                              cv::Mat proto(proto_shape, CV_32FC1, const_cast<float*>(proto_data));

                              const std::vector<int> mask_shape0({n, mask_dim});
                              cv::Mat mask(mask_shape0, CV_32FC1, mask_fuse0.data());

                              mask *= proto;
                              cv::exp(-mask, mask);
                              mask = 1 / (1 + mask);

                              const std::vector<int> mask_shape1({n, mask_h, mask_w});    // [2, 160, 160]
                              mask = mask.reshape(1, mask_shape1);

                              const float w_ratio = static_cast<float>(mask_w) / static_cast<float>(itensor_size_.width);
                              const float h_ratio = static_cast<float>(mask_h) / static_cast<float>(itensor_size_.height);
                              std::vector<NNOutput>& box = boxes[bs_id];      // Container used to store current batch result

                              const std::vector<int> mask_shape2({mask_h, mask_w});    // [160, 160]
                              for (int i = 0; i < n; ++i) {
                                  const std::vector<cv::Range> n_ranges({cv::Range(i, i + 1), cv::Range::all(), cv::Range::all()});
                                  cv::Mat mask_ = mask(n_ranges).clone();
                                  mask_ = mask_.reshape(1, mask_shape2);
                                  cv::Mat roi_mask = mask_.clone();
                                  const int downsampled_bbox_x = std::clamp(static_cast<int>(rects2_f[i].x * w_ratio), 0, mask_w - 1);
                                  const int downsampled_bbox_y = std::clamp(static_cast<int>(rects2_f[i].y * h_ratio), 0, mask_h - 1);
                                  const int bbox_w = static_cast<int>(rects2_f[i].width * w_ratio);
                                  const int bbox_h = static_cast<int>(rects2_f[i].height * h_ratio);
                                  if (mask_w - downsampled_bbox_x - 1 < 1 || mask_h - downsampled_bbox_y - 1 < 1)       continue;
                                  const int downsampled_bbox_w = std::clamp(
                                          bbox_w, 1, mask_w - downsampled_bbox_x - 1);
                                  const int downsampled_bbox_h = std::clamp(
                                          bbox_h, 1, mask_h - downsampled_bbox_y - 1);

                                  cv::Rect downsampled_bboxes({downsampled_bbox_x, downsampled_bbox_y,
                                                               downsampled_bbox_w, downsampled_bbox_h});
                                  roi_mask(downsampled_bboxes).setTo(0);
                                  mask_ -= roi_mask;

                                  cv::resize(mask_, mask_, itensor_size_);
                                  ScaleImage(&mask_, itensor_size_, img_size);
                                  cv::threshold(mask_, mask_, 0.5, 1.0, cv::THRESH_BINARY);
                                  mask_.convertTo(mask_, CV_8UC1);

                                  box.emplace_back(confs2[i], ids2[i], ScaleCoords(rects2_f[i], itensor_size_, img_size),
                                                   mask_, class_names_[ids2[i]]);
                              }
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
        size_t vector_num_;       // 25200
        size_t vector_length_;    // 117: 80 + 5 + 32
        size_t mask_dim_;      // 32
        size_t mask_h_;        // 160
        size_t mask_w_;        // 160
        size_t num_classes_;
        cv::Size itensor_size_;

        std::shared_ptr<Inference> infer_;
    public:
        ~YoloV5_Seg() noexcept = default;

        YoloV5_Seg(YoloV5_Seg&&) noexcept = default;

        YoloV5_Seg& operator=(YoloV5_Seg&&) noexcept = default;

        YoloV5_Seg(const YoloV5_Seg&) noexcept = delete;

        YoloV5_Seg& operator=(const YoloV5_Seg&) noexcept = delete;

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

            const std::vector<std::string> input_names = {"images"};
            const std::vector<std::string> output_names = {"output0", "output1"};

            cv::Mat tensor = Preprocess(i_tensors);

//            std::cout << "\nInput: " << std::endl;
//            std::vector<cv::Range> range0({cv::Range::all(), cv::Range(0, 1), cv::Range::all(), cv::Range::all()});
//            std::cout << "Input ch0 sum: " << std::setprecision(10) << cv::sum(tensor(range0))[0] << std::endl;
//
//            std::vector<cv::Range> range1({cv::Range::all(), cv::Range(1, 2), cv::Range::all(), cv::Range::all()});
//            std::cout << "Input ch1 sum: " << std::setprecision(10) << cv::sum(tensor(range1))[0] << std::endl;
//
//            std::vector<cv::Range> range2({cv::Range::all(), cv::Range(2, 3), cv::Range::all(), cv::Range::all()});
//            std::cout << "Input ch2 sum: " << std::setprecision(10) << cv::sum(tensor(range2))[0] << std::endl;

            const float *iptr = (float*)tensor.data;

            std::vector<float> output0(bs_ * vector_num_ * vector_length_);
            std::vector<float> output1(bs_ * mask_dim_ * mask_h_ * mask_w_);

            std::vector<float*> outputs({output0.data(), output1.data()});

            err = infer_->Run(&iptr, input_names, outputs.data(), output_names);
            if (err) {
                *ec = err;
                return {};
            }

//            std::cout << "\nOuts: " << std::endl;
//            float out_sum = .0f;
//            for (auto t: output0)   out_sum += t;
//            std::cout << "pred: " << std::setprecision(15) << out_sum << std::endl;
//
//            out_sum = .0f;
//            for (auto t: output1)   out_sum += t;
//            std::cout << "proto: " << std::setprecision(15) << out_sum << std::endl;

            return Decode(const_cast<const float**>(outputs.data()), i_tensors);
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
                const auto &mask = output.mask;
                if (mask.empty())       continue;
                //! Draw the mask on the image.
                cv::add(draw_img, colors[i] * 0.5, draw_img, mask);
            }

            for (size_t i = 0; i < outputs.size(); ++i) {
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

        static std::shared_ptr<YoloV5_Seg> CreateModel(Error* ec, std::shared_ptr<Inference> infer) noexcept {
            using YoloV5_Seg_ = YoloV5_Seg<Inference>;
            std::shared_ptr<YoloV5_Seg_> yolov5_seg(new YoloV5_Seg_());

            Error err = yolov5_seg->CheckAndSetInfer(infer);
            if (err) {
                *ec = err;

                return nullptr;
            }

            return yolov5_seg;
        }
    };
}

#endif //AI_TEMPLATE_YOLOV5_SEG_H
