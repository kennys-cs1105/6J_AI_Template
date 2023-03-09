//
// Created by zyh on 11/24/22.
//

#ifndef AI_TEMPLATE_TRT_H
#define AI_TEMPLATE_TRT_H

#include "../include/inference.h"

namespace nn {
    class TensorRTImpl final : public TensorRT {
    public:
        // Ctor
        TensorRTImpl() = default;

        // Dtor
        ~TensorRTImpl() noexcept final = default;

        // Move ctor
        TensorRTImpl(TensorRTImpl &&) noexcept = default;

        // Move assignment operator
        TensorRTImpl &operator=(TensorRTImpl&&) noexcept = default;

        // Copy ctor
        TensorRTImpl(const TensorRTImpl&) = delete;

        // Copy assignment operator
        TensorRTImpl &operator=(const TensorRTImpl &) = delete;

        void SetTensorInfo(const nvinfer1::ICudaEngine&);

        Error Create(const std::unordered_map<std::uint32_t, std::uint32_t>& cuda_device_concurrency,
                     const std::vector<char>& engine_data, const std::string& name);

        Error Run(const float** inputs, const std::vector<std::string> &input_names,
                  float** outputs, const std::vector<std::string> &output_names) final;

        std::string Print() final;

        const std::vector<TensorInfo>& GetInputInfo() final;

        const std::vector<TensorInfo>& GetOutputInfo() final;

        std::optional<TensorInfo> GetInputInfoByName(const std::string &name) final;

        std::optional<TensorInfo> GetOutputInfoByName(const std::string &name) final;

        static std::uint64_t GetRandomUInt64(std::uint64_t begin, std::uint64_t end);

        static Error CheckPath(const std::string& path);

        static Error CheckCudaDevice(const std::unordered_map<std::uint32_t, std::uint32_t>& cuda_device_concurrency_);

    private:
        class TRTLogger final : public nvinfer1::ILogger {
        public:
            TRTLogger() = default;
            ~TRTLogger() final = default;
            static nvinfer1::ILogger& GetInstance() noexcept;
            void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept final;
        };

        class TRTContext {
        public:
            TRTContext() : context_(nullptr), stream_(nullptr) {}

            ~TRTContext();

            TRTContext(TRTContext&& trt_ctx) noexcept;

            TRTContext &operator=(TRTContext&&) noexcept;

            TRTContext(const TRTContext&) = delete;

            TRTContext &operator=(const TRTContext &) = delete;

            Error Create(nvinfer1::ICudaEngine &engine);

            Error Run(const float **inputs, const std::vector<std::string> &input_names,
                                    float **outputs, const std::vector<std::string> &output_names,
                                    const nvinfer1::ICudaEngine& engine);

        private:
            std::shared_ptr<nvinfer1::IExecutionContext> context_;
            std::vector<void *> cuda_buffers_;
            cudaStream_t stream_;
            std::mutex mutex_;
        };

        class TRTExecutorOnDevice {
        public:
            TRTExecutorOnDevice() = default;

            ~TRTExecutorOnDevice() = default;

            TRTExecutorOnDevice(TRTExecutorOnDevice&&) noexcept = default;

            TRTExecutorOnDevice &operator=(TRTExecutorOnDevice&&) noexcept = default;

            TRTExecutorOnDevice(const TRTExecutorOnDevice&) = delete;

            TRTExecutorOnDevice &operator=(const TRTExecutorOnDevice &) = delete;

            Error Create(std::shared_ptr<nvinfer1::ICudaEngine> eng, std::uint32_t concurrency);

            Error Run(const float** inputs, const std::vector<std::string> &input_names,
                                    float** outputs, const std::vector<std::string> &output_names);

        private:
            std::shared_ptr<nvinfer1::ICudaEngine> engine_;
            std::vector<TRTContext> contexts_;
        };

        std::unordered_map<std::uint32_t, TRTExecutorOnDevice> id_exe_;
        std::vector<TensorInfo> inputs_info_;
        std::vector<TensorInfo> outputs_info_;
        std::string name_;
    };

}




#endif //AI_TEMPLATE_TRT_H

