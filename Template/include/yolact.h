//
// Created by zyh on 12/14/22.
//

#ifndef AI_TEMPLATE_YOLACT_H
#define AI_TEMPLATE_YOLACT_H

#include <utility>
#include <opencv2/opencv.hpp>
#include <random>

namespace nn {
    //! Yolact: A instance segmentation net.
    template <typename Inference>
    class Yolact {
    private:
        //! Default constructor of Class Yolact.
        Yolact() noexcept : infer_(nullptr), nms_threshold_(.5f), cc_nms_threshold_(.99f),
            bs_(0), vector_num_(0), num_classes_(0), mask_size_(0), proto_wh_(0) {}

        //! Check input is valid or not.
        Error CheckInput(const std::vector<cv::Mat>& images) noexcept {
            if (bs_ != images.size())    return Error::ErrorInvalidInput;
            for (const auto& img: images)
                if (img.empty() || ((img.channels() != 3) && (img.channels() != 1)) ||
                    img.rows < 20 || img.cols < 20)
                    return Error::ErrorInvalidInput;

            return Success;
        }

        //! Check inference engine is valid or not.
        //! If it's ok, set some crucial parameters to this Yolact object.
        //! Otherwise, return an error code.
        Error CheckAndSetInfer(std::shared_ptr<Inference> infer) noexcept {
            //! Check inputs info.
            const auto& inputs_info = infer->GetInputInfo();
            if (inputs_info.size() != 1)    return Error::ErrorInvalidModelFile;

            const auto images_info = infer->GetInputInfoByName("images");
            const auto &i_shape = images_info->shape;
            if (!images_info.has_value() || i_shape.size() != 4)
                return Error::ErrorInvalidModelFile;
            itensor_size_ = {static_cast<int>(i_shape[3]), static_cast<int>(i_shape[2])};

            //! Check outputs info.
            const auto& outputs_info = infer->GetOutputInfo();
            if (outputs_info.size() != 5)   return Error::ErrorInvalidModelFile;

            //! Get the loc info by name and check it.
            const auto loc_info = infer->GetOutputInfoByName("loc");
            if (!loc_info.has_value())      return Error::ErrorInvalidModelFile;

            const auto& loc_shape = loc_info->shape;
            if (loc_shape.size() != 3)      return Error::ErrorInvalidModelFile;

            if (loc_shape[0] <= 0 || loc_shape[1] <= 0 || loc_shape[2] != 4)
                return Error::ErrorInvalidModelFile;
            bs_ = static_cast<size_t>(loc_shape[0]);
            vector_num_ = static_cast<size_t>(loc_shape[1]);

            //! Get the conf info by name and check it.
            const auto conf_info = infer->GetOutputInfoByName("conf");
            if (!conf_info.has_value())      return Error::ErrorInvalidModelFile;

            const auto &conf_shape = conf_info->shape;
            if (conf_shape.size() != 3 || conf_shape[0] != bs_ || conf_shape[1] != vector_num_ || conf_shape[2] < 2)
                return Error::ErrorInvalidModelFile;
            num_classes_ = static_cast<size_t>(conf_shape[2] - 1);

            //! Get the mask info by name and check it.
            const auto mask_info = infer->GetOutputInfoByName("mask");
            if (!mask_info.has_value())      return Error::ErrorInvalidModelFile;

            const auto &mask_shape = mask_info->shape;
            if (mask_shape.size() != 3 || mask_shape[0] != bs_ || mask_shape[1] != vector_num_ || mask_shape[2] <= 0)
                return Error::ErrorInvalidModelFile;
            mask_size_ = static_cast<size_t>(mask_shape[2]);

            //! Get the prior info by name and check it.
            const auto prior_info = infer->GetOutputInfoByName("prior");
            if (!prior_info.has_value())      return Error::ErrorInvalidModelFile;

            const auto &prior_shape = prior_info->shape;
            if (prior_shape.size() != 2 || prior_shape[0] != vector_num_ || prior_shape[1] != 4)
                return Error::ErrorInvalidModelFile;

            //! Get the proto info by name and check it.
            const auto proto_info = infer->GetOutputInfoByName("proto");
            if (!proto_info.has_value())      return Error::ErrorInvalidModelFile;

            const auto &proto_shape = proto_info->shape;
            if (proto_shape.size() != 4 || proto_shape[0] != bs_ || proto_shape[3] != mask_size_ ||
                proto_shape[1] != proto_shape[2] || proto_shape[1] <= 0)
                return Error::ErrorInvalidModelFile;
            proto_wh_ = static_cast<size_t>(proto_shape[1]);

            infer_ = infer;

            return Error::Success;
        }

        //! Do preprocess for Yolact:
        //! 1. Convert data type to float32.
        //! 2. Resize the image to the size of the input tensor.
        //! 3. Normalize.
        //! The cv::Mat returned has this shape: [batch_size, 3, h, w]
        cv::Mat Preprocess(const std::vector<cv::Mat>& src) noexcept {
            std::vector<cv::Mat> new_src;
            for (size_t i = 0; i < src.size(); ++i) {
                new_src.emplace_back(src[i].clone());
                new_src[i].convertTo(new_src[i], CV_32F);
                if (new_src[i].channels() == 1)     cv::cvtColor(new_src[i], new_src[i], cv::COLOR_GRAY2BGR);
                cv::resize(new_src[i], new_src[i], itensor_size_);
                new_src[i] -= means_;
                new_src[i] /= stds_;
            }

            //! This function will return a 4-d cv::Mat created by images in a std::vector.
            return cv::dnn::blobFromImages(new_src, 1.0f, itensor_size_, cv::Scalar(), true);
        }

        //! Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        //! Also converts from relative to absolute coordinates based on the original image.
        cv::Rect SanitizeCoordinates(const cv::Rect2f& box, const cv::Size &size, int padding) {
            double xa = box.x * static_cast<double>(size.width), ya = box.y * static_cast<double>(size.height);
            double xb = (box.x + box.width) * static_cast<double>(size.width),
                   yb = (box.y + box.height) * static_cast<double>(size.height);

            //! Padding value can be set.
            double x0 = std::min(xa, xb) - padding, y0 = std::min(ya, yb) - padding;
            double x1 = std::max(xa, xb) + padding, y1 = std::max(ya, yb) + padding;

            //! Make sure that x0_, y0_, x1_, y1_ are valid.
            int x0_ = std::clamp(static_cast<int>(x0), 0, size.width);
            int y0_ = std::clamp(static_cast<int>(y0), 0, size.height);
            int x1_ = std::clamp(static_cast<int>(x1), 0, size.width);
            int y1_ = std::clamp(static_cast<int>(y1), 0, size.height);

            //! Notice! Class cv::Rect 's constructor just receives x, y, w, and h.
            return {x0_, y0_, x1_ - x0_, y1_ - y0_};
        }

        //! Decode outputs to segmentation result.
        std::vector<std::vector<NNOutput>> Decode(const float **tensor, const std::vector<cv::Mat>& src) noexcept {
            std::vector<std::vector<NNOutput>> boxes(bs_);   // bs, n
            std::vector<size_t> bs_idx(bs_, 0);
            std::iota(bs_idx.begin(), bs_idx.end(), 0);

            // Loop for batches.
            std::for_each(bs_idx.begin(), bs_idx.end(),
                          [tensor, &boxes, &src, this] (const size_t bs_id) {
                const auto& img_size = src[bs_id].size();

                const float* loc_data = tensor[0] + bs_id * vector_num_ * 4;
                const float* conf_data = tensor[1] + bs_id * vector_num_ * (num_classes_ + 1);
                const float* mask_data = tensor[2] + bs_id * vector_num_ * mask_size_;
                const float* prior_data = tensor[3];
                std::vector<const float*> data_ptrs({loc_data, conf_data, mask_data, prior_data});

                std::vector<cv::Rect2f> rects0_f;
                std::vector<cv::Rect> offset_rects;
                std::vector<float> confs0;
                std::vector<std::int64_t> ids0;
                std::vector<std::vector<float>> masks0;
                std::vector<void*> res_ptrs({&rects0_f, &offset_rects, &confs0, &ids0, &masks0});

                std::vector<size_t> vector_idx(vector_num_, 0);
                std::iota(vector_idx.begin(), vector_idx.end(), 0);

                // Loop for vectors.
                std::for_each(vector_idx.begin(), vector_idx.end(),
                              [&data_ptrs, &res_ptrs, this] (const size_t& vector_id) {
                    // Parse confidence and class id.
                    const float* conf_vec = data_ptrs[1] + vector_id * (num_classes_ + 1);
                    const float* max_conf = std::max_element(conf_vec + 1, conf_vec + num_classes_ + 1);
                    std::int64_t class_id = max_conf - (conf_vec + 1);

                    if (*max_conf < cls_thresholds_[class_id])  return;

                    static_cast<std::vector<float>*>(res_ptrs[2])->emplace_back(*max_conf);
                    static_cast<std::vector<std::int64_t>*>(res_ptrs[3])->emplace_back(class_id);

                    // Parse bounding box.
                    const float* loc_vec = data_ptrs[0] + vector_id * 4;
                    const float* prior_vec = data_ptrs[3] + vector_id * 4;
                    float center_x = prior_vec[0] + loc_vec[0] * variances_[0] * prior_vec[2];
                    float center_y = prior_vec[1] + loc_vec[1] * variances_[0] * prior_vec[3];
                    float w = prior_vec[2] * std::exp(loc_vec[2] * variances_[1]);
                    float h = prior_vec[3] * std::exp(loc_vec[3] * variances_[1]);

                    cv::Rect2f bbox_ori({center_x - w / 2.0f, center_y - h / 2.0f, w, h});
                    cv::Rect bbox_nms(SanitizeCoordinates(bbox_ori, itensor_size_, 0));

                    static_cast<std::vector<cv::Rect2f>*>(res_ptrs[0])->emplace_back(bbox_ori);
                    static_cast<std::vector<cv::Rect>*>(res_ptrs[1])->emplace_back(
                            bbox_nms.x + 4096 * class_id, bbox_nms.y + 4096 * class_id,
                            bbox_nms.width, bbox_nms.height);

                    // Parse mask.
                    const float* mask_vec = data_ptrs[2] + vector_id * mask_size_;
                    static_cast<std::vector<std::vector<float>>*>(res_ptrs[4])->emplace_back(mask_vec, mask_vec + mask_size_);
                });

                if (offset_rects.empty())   return;

                std::vector<int> indices;
                cv::dnn::NMSBoxes(offset_rects, confs0, 0, nms_threshold_, indices);

                std::vector<cv::Rect2f> rects1_f;
                std::vector<cv::Rect> rects0;
                std::vector<float> confs1;
                std::vector<std::int64_t> ids1;
                std::vector<std::vector<float>> masks1;
                for (const int ix : indices) {
                    rects1_f.emplace_back(rects0_f[ix]);
                    rects0.emplace_back(SanitizeCoordinates(rects0_f[ix], img_size, 0));
                    confs1.emplace_back(confs0[ix]);
                    ids1.emplace_back(ids0[ix]);
                    masks1.emplace_back(std::move(masks0[ix]));
                }

                indices.clear();
                cv::dnn::NMSBoxes(rects0, confs1, 0, cc_nms_threshold_, indices);

                std::vector<cv::Rect2f> rects2_f;
                std::vector<cv::Rect> rects1;
                std::vector<float> confs2;
                std::vector<std::int64_t> ids2;
                std::vector<float> mask_fuse0;
                for (const int ix : indices) {
                    rects2_f.emplace_back(rects1_f[ix]);
                    rects1.emplace_back(rects0[ix]);
                    confs2.emplace_back(confs1[ix]);
                    ids2.emplace_back(ids1[ix]);
                    mask_fuse0.insert(mask_fuse0.end(), masks1[ix].begin(), masks1[ix].end());
                }

                // Generate mask with proto.
                const int proto_wh = static_cast<int>(proto_wh_);
                const int mask_size = static_cast<int>(mask_size_);

                const float* proto_data = tensor[4] + bs_id * proto_wh_ * proto_wh_ * mask_size_;
                const std::vector<int> proto_shape0({proto_wh, proto_wh, mask_size});   // [138, 138, 32]
                const int n = static_cast<int>(rects2_f.size());
                cv::Mat proto(proto_shape0, CV_32FC1, const_cast<float*>(proto_data));

                const std::vector<int> proto_shape1({proto_wh * proto_wh, mask_size});  // [19044, 32]
                proto = proto.reshape(1, proto_shape1);

                const std::vector<int> mask_shape0({n, proto_shape1[1]});       // [n, 32]
                cv::Mat mask_fuse1(mask_shape0, CV_32FC1, mask_fuse0.data());
                mask_fuse1 = mask_fuse1.t();                                    // [32, n]

                mask_fuse1 = proto * mask_fuse1;                                // [19044, 32] * [32, n]

                const std::vector<int> mask_shape1({proto_wh, proto_wh, n});    // [138, 138, n]
                mask_fuse1 = mask_fuse1.reshape(1, mask_shape1);

                cv::exp(-mask_fuse1, mask_fuse1);
                mask_fuse1 = 1 / (1 + mask_fuse1);

                const std::vector<int> mask_shape2({proto_wh, proto_wh});
                std::vector<NNOutput>& box = boxes[bs_id];
                for (int i = 0; i < n; ++i) {
                    const std::vector<cv::Range> ranges({cv::Range::all(), cv::Range::all(), cv::Range(i, i + 1)});
                    cv::Mat mask = mask_fuse1(ranges).clone();
                    mask = mask.reshape(1, mask_shape2);

                    const cv::Rect2f& rect_f = rects2_f[i];
                    cv::Rect rect_dst = SanitizeCoordinates(rect_f, cv::Size(proto_wh, proto_wh), 1);
                    const std::vector<cv::Range> roi_ranges({cv::Range(rect_dst.y, rect_dst.y + rect_dst.height),
                                                             cv::Range(rect_dst.x, rect_dst.x + rect_dst.width)});
                    cv::Mat roi_mask = mask.clone();
                    roi_mask(roi_ranges).setTo(0);
                    mask -= roi_mask;

                    // This will be changed to the same as YoloV5 post-processing in the upgrades.
                    cv::resize(mask, mask, img_size);

                    cv::threshold(mask, mask, 0.5, 1.0, cv::THRESH_BINARY);
                    mask.convertTo(mask, CV_8UC1);

                    box.emplace_back(confs2[i], ids2[i], rects1[i], mask, class_names_[ids2[i]]);
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
        size_t vector_num_;
        size_t num_classes_;
        size_t mask_size_;
        size_t proto_wh_;

        cv::Size itensor_size_;
        cv::Scalar means_;
        cv::Scalar stds_;
        std::vector<float> variances_;

        std::shared_ptr<Inference> infer_;
    public:
        ~Yolact() = default;

        Yolact(Yolact&&) noexcept = default;

        Yolact& operator=(Yolact&&) noexcept = default;

        Yolact(const Yolact&) noexcept = delete;

        Yolact& operator=(const Yolact&) noexcept = delete;

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
            const std::vector<std::string> output_names = {"loc", "conf", "mask", "prior", "proto"};

            cv::Mat tensor = Preprocess(i_tensors);

            const float *iptr = (float*)tensor.data;

            std::vector<float> loc_outputs(bs_ * vector_num_ * 4);
            std::vector<float> conf_outputs(bs_ * vector_num_ * (num_classes_ + 1));
            std::vector<float> mask_outputs(bs_ * vector_num_ * mask_size_);
            std::vector<float> prior_outputs(vector_num_ * 4);
            std::vector<float> proto_outputs(bs_ * proto_wh_ * proto_wh_ * mask_size_);

            std::vector<float*> outputs({loc_outputs.data(), conf_outputs.data(),
                         mask_outputs.data(), prior_outputs.data(), proto_outputs.data()});

            err = infer_->Run(&iptr, input_names, outputs.data(), output_names);
            if (err) {
                *ec = err;
                return {};
            }

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

        static std::shared_ptr<Yolact> CreateModel(Error* ec, std::shared_ptr<Inference> infer) noexcept {
            using Yolact_ = Yolact<Inference>;
            std::shared_ptr<Yolact_> yolact_(new Yolact_());

            Error err = yolact_->CheckAndSetInfer(infer);
            if (err) {
                *ec = err;

                return nullptr;
            }

            // These correspond to BGR.
            yolact_->means_ = {103.94, 116.78, 123.68};
            yolact_->stds_ = {57.38, 57.12, 58.40};

            yolact_->variances_ = {0.1, 0.2};

            return yolact_;
        }
    };
}

#endif //AI_TEMPLATE_YOLACT_H
