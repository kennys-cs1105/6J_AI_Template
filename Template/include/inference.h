//
// Created by zyh on 11/24/22.
//

#ifndef AI_TEMPLATE_ENGINE_H
#define AI_TEMPLATE_ENGINE_H

#include <optional>
#include <memory>
#include <unordered_map>
#include <thread>
#include <vector>
#include "base.h"

namespace nn {
    class TensorRT {
    private:
        class TensorRTConfig {
        private:
            std::unordered_map<std::uint32_t, std::uint32_t> cuda_device_concurrency_;
        public:
            TensorRTConfig(std::string path, std::string name) :
                    path(std::move(path)), name(std::move(name)),
                    cuda_device_concurrency_({std::pair<std::uint32_t, std::uint32_t>(0, 1)}) {}

            const std::string path;
            const std::string name;

            void SetCudaDeviceConcurrency(std::uint32_t device_id, std::uint32_t concurrency) {
                cuda_device_concurrency_[device_id] = concurrency;
            }

            const std::unordered_map<std::uint32_t, std::uint32_t> &GetCudaDeviceConcurrency() const {
                return cuda_device_concurrency_;
            }
        };
    public:
        virtual ~TensorRT() = default;

        TensorRT(TensorRT &&) = default;

        TensorRT(const TensorRT &) = delete;

        TensorRT &operator=(const TensorRT &) = delete;

        using Config = TensorRTConfig;
        using TensorInfo = TensorInfo_<int>;

        static std::shared_ptr<TensorRT> MakeEngine(Error* ec, const Config& config);

        virtual Error Run(const float **inputs, const std::vector<std::string> &input_names,
                          float **outputs, const std::vector<std::string> &output_names) = 0;

        virtual const std::vector<TensorInfo> &GetInputInfo() = 0;

        virtual const std::vector<TensorInfo> &GetOutputInfo() = 0;

        virtual std::optional<TensorInfo> GetInputInfoByName(const std::string &name) = 0;

        virtual std::optional<TensorInfo> GetOutputInfoByName(const std::string &name) = 0;

        virtual std::string Print() = 0;

    protected:
        TensorRT() = default;
    };

    class OnnxRT {
    private:
        class OnnxRTConfig {
        public:
            OnnxRTConfig(std::string path, std::string name,
                         std::uint32_t conc = 1,
                         std::uint32_t intra = 0,
                         std::uint32_t inter = 0) :
                    path(std::move(path)), name(std::move(name)),
                    concurrency(conc), intra_op_threads(intra), inter_op_threads(inter) {}

            const std::string path;
            const std::string name;
            const std::uint32_t concurrency;
            const std::uint32_t intra_op_threads;
            const std::uint32_t inter_op_threads;
        };
    public:
        virtual ~OnnxRT() = default;    // 虚dtor析构函数

        OnnxRT(OnnxRT &&) = default;    // 默认mtor移动构造函数

        OnnxRT &operator=(OnnxRT&&) = delete;  // 禁止mv assignment移动赋值函数

        OnnxRT(const OnnxRT &) = delete;    // 禁止cptor拷贝构造函数

        OnnxRT &operator=(const OnnxRT &) = delete;     // 禁止cp assignment拷贝赋值函数

        typedef OnnxRTConfig Config;
        typedef TensorInfo_<std::int64_t> TensorInfo;

        static std::shared_ptr<OnnxRT> MakeEngine(Error* ec, const Config& config);

        virtual Error Run(const float **inputs, const std::vector<std::string> &input_names,
                          float **outputs, const std::vector<std::string> &output_names) = 0;

        virtual const std::vector<TensorInfo> &GetInputInfo() = 0;

        virtual const std::vector<TensorInfo> &GetOutputInfo() = 0;

        virtual std::optional<TensorInfo> GetInputInfoByName(const std::string &name) = 0;

        virtual std::optional<TensorInfo> GetOutputInfoByName(const std::string &name) = 0;

        virtual std::string Print() = 0;

    protected:
        OnnxRT() = default; // 不公开的ctor
    };
}













#endif //AI_TEMPLATE_ENGINE_H
