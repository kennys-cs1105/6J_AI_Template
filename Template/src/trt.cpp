//
// Created by zyh on 11/24/22.
//

#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include <numeric>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <utility>
#include "trt.h"

namespace nn {
    void TensorRTImpl::TRTLogger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
        if (severity <= Severity::kWARNING)
            std::cerr << msg << std::endl;
    }

    nvinfer1::ILogger &TensorRTImpl::TRTLogger::GetInstance() noexcept {
        static TRTLogger logger;

        return logger;
    }

    TensorRTImpl::TRTContext::TRTContext(TensorRTImpl::TRTContext &&trt_ctx) noexcept {
        stream_ = trt_ctx.stream_;
        trt_ctx.stream_ = nullptr;

        context_ = std::move(trt_ctx.context_);
        cuda_buffers_ = std::move(trt_ctx.cuda_buffers_);
    }

    TensorRTImpl::TRTContext &TensorRTImpl::TRTContext::operator=(TensorRTImpl::TRTContext &&trt_ctx) noexcept {
        if (stream_)
            cudaStreamDestroy(stream_);
        for (void* p : cuda_buffers_)
            cudaFree(p);

        stream_ = trt_ctx.stream_;
        trt_ctx.stream_ = nullptr;

        context_ = std::move(trt_ctx.context_);
        cuda_buffers_ = std::move(trt_ctx.cuda_buffers_);

        return *this;
    }

    TensorRTImpl::TRTContext::~TRTContext() {
        if (stream_)
            cudaStreamDestroy(stream_);
        for (void* p : cuda_buffers_)
            cudaFree(p);
    }

    Error TensorRTImpl::TRTContext::Create(nvinfer1::ICudaEngine &engine) {
        context_.reset(engine.createExecutionContext());
        if (!context_ || cudaStreamCreate(&stream_))
            return Error::ErrorInferEnvError;

        int buffer_num = engine.getNbBindings();
        size_t mb = 0;

        for (int i = 0; i < buffer_num; i++) {
            void* buffer = nullptr;
            nvinfer1::Dims dims = engine.getBindingDimensions(i);
            size_t count = accumulate(
                    dims.d, dims.d + dims.nbDims, 1, std::multiplies<>());

            if (cudaMalloc(&buffer, count * sizeof(float)))
                return Error::ErrorInferEnvError;

            mb += count * sizeof(float) / 1024 / 1024;
            cuda_buffers_.emplace_back(buffer);
        }

        return Error::Success;
    }

    Error TensorRTImpl::TRTContext::Run(const float **inputs, const std::vector<std::string> &input_names,
                          float **outputs, const std::vector<std::string> &output_names,
                          const nvinfer1::ICudaEngine& engine) {
        int buffer_num = engine.getNbBindings();
        std::vector<bool> is_input(buffer_num, false);
        std::vector<size_t> counts(buffer_num, 0);
        std::vector<size_t> ids(buffer_num, 0);

        for (int i = 0; i < buffer_num; ++i) {
            nvinfer1::Dims dims = engine.getBindingDimensions(i);
            size_t count = accumulate(
                    dims.d, dims.d + dims.nbDims, 1, std::multiplies<>());
            const std::string binding_name = engine.getBindingName(i);
            if (engine.bindingIsInput(i)) {
                is_input[i] = true;
                auto iter = std::find(input_names.begin(), input_names.end(), binding_name);
                if (iter == input_names.end())
                    return Error::ErrorInvalidTensor;
                size_t idx = iter - input_names.begin();
                ids[i] = idx;
                counts[i] = count;
            } else {
                auto iter = std::find(output_names.begin(), output_names.end(), binding_name);
                if (iter == output_names.end())
                    continue;
                size_t idx = iter - output_names.begin();
                ids[i] = idx;
                counts[i] = count;
            }
        }

        cudaEvent_t end = nullptr;
        cudaEventCreateWithFlags(&end, cudaEventBlockingSync);

        {
            std::lock_guard<std::mutex> lg(mutex_);
            for (int i = 0; i < buffer_num; ++i)
                if (is_input[i]) {
                    if (cudaMemcpyAsync(cuda_buffers_[i], static_cast<const void *>(inputs[ids[i]]),
                                        counts[i] * sizeof(float), cudaMemcpyHostToDevice, stream_)) {
                        cudaEventDestroy(end);

                        return Error::ErrorInferenceError;
                    }
                }

            if (!context_->enqueueV2(cuda_buffers_.data(), stream_, nullptr))
                return Error::ErrorInferenceError;

            for (int i = 0; i < buffer_num; ++i)
                if (!is_input[i] && counts[i]) {
                    if (cudaMemcpyAsync(static_cast<void*>(outputs[ids[i]]), cuda_buffers_[i],
                                        counts[i] * sizeof(float), cudaMemcpyDeviceToHost, stream_)) {
                        cudaEventDestroy(end);

                        return Error::ErrorInferenceError;
                    }
                }
        }

        cudaEventRecord(end, stream_);
        cudaEventSynchronize(end);
        cudaEventDestroy(end);

        return Error::Success;
    }

    Error TensorRTImpl::TRTExecutorOnDevice::Create(std::shared_ptr<nvinfer1::ICudaEngine> eng, std::uint32_t concurrency) {
        engine_ = std::move(eng);
        for (int i = 0; i < concurrency; ++i) {
            TRTContext ctx;
            Error err = ctx.Create(*engine_);
            if (err)    return err;
            contexts_.emplace_back(std::move(ctx));
        }

        return Error::Success;
    }

    Error TensorRTImpl::TRTExecutorOnDevice::Run(const float **inputs, const std::vector<std::string> &input_names,
                                   float **outputs, const std::vector<std::string> &output_names) {
        size_t context_id = TensorRTImpl::GetRandomUInt64(0, contexts_.size() - 1);

        Error ec = contexts_[context_id].Run(inputs, input_names, outputs, output_names, *engine_);
        if (ec)     return ec;

        return Error::Success;
    }

    void TensorRTImpl::SetTensorInfo(const nvinfer1::ICudaEngine &engine) {
        int buffer_num = engine.getNbBindings();
        for (int i = 0; i < buffer_num; i++) {
            std::string binding_name = engine.getBindingName(i);
            nvinfer1::Dims dims = engine.getBindingDimensions(i);
            std::vector<int> buffer_dims;

            for (int j = 0; j < dims.nbDims; j++)
                buffer_dims.emplace_back(dims.d[j]);

            if (engine.bindingIsInput(i))
                inputs_info_.emplace_back(binding_name, buffer_dims);
            else
                outputs_info_.emplace_back(binding_name, buffer_dims);
        }
    }

    Error TensorRTImpl::Create(const std::unordered_map<std::uint32_t, std::uint32_t> &cuda_device_concurrency,
                           const std::vector<char> &engine_data, const std::string& name) {
        Error err = Error::Success;
        this->name_ = name;
        nvinfer1::ILogger& logger = TRTLogger::GetInstance();
        static std::once_flag flag;
        std::call_once(flag, [&err, &logger] {
            if (!initLibNvInferPlugins(&logger, "TRT"))
                err = Error::ErrorInferEnvError;
        });

        if (err)    return err;

        static std::shared_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime)       return Error::ErrorInferEnvError;

        for (auto &m : cuda_device_concurrency) {
            const std::uint32_t id = std::get<0>(m);

            if (cudaSetDevice(static_cast<int>(id)))
                return Error::ErrorInferEnvError;

            std::shared_ptr<nvinfer1::ICudaEngine> engine(
                    runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
            if (!engine)    return Error::ErrorInvalidModelFile;

            if (GetInputInfo().empty() && GetOutputInfo().empty())
                SetTensorInfo(*engine);

            const std::uint32_t concurrency = std::get<1>(m);
            TRTExecutorOnDevice exe;
            Error err = exe.Create(engine, concurrency);
            if (err)    return err;

            id_exe_.emplace(id, std::move(exe));
        }

        return Error::Success;
    }

    Error TensorRTImpl::Run(const float **inputs, const std::vector<std::string> &input_names,
                        float **outputs, const std::vector<std::string> &output_names) {
        if (input_names.size() != inputs_info_.size())
            return Error::ErrorInvalidTensor;
        if (output_names.size() != outputs_info_.size())
            return Error::ErrorInvalidTensor;

        std::vector<std::uint32_t> devices;
        for (const auto& pair : id_exe_)
            devices.emplace_back(std::get<0>(pair));

        std::uint32_t device_id = devices[GetRandomUInt64(0, devices.size() - 1)];

        Error ec = id_exe_[device_id].Run(inputs, input_names, outputs, output_names);
        if (ec)         return ec;

        return Error::Success;
    }

    const std::vector<TensorRTImpl::TensorInfo> &TensorRTImpl::GetInputInfo() {
        return inputs_info_;
    }

    const std::vector<TensorRTImpl::TensorInfo> &TensorRTImpl::GetOutputInfo() {
        return outputs_info_;
    }

    std::optional<TensorRTImpl::TensorInfo> TensorRTImpl::GetInputInfoByName(const std::string &name) {
        std::optional<TensorRTImpl::TensorInfo> in;
        for (auto &info: inputs_info_)
            if (info.name == name) {
                in.emplace(info);
                break;
            }

        return in;
    }

    std::optional<TensorRTImpl::TensorInfo> TensorRTImpl::GetOutputInfoByName(const std::string &name) {
        std::optional<TensorRTImpl::TensorInfo> out;
        for (auto &info: outputs_info_)
            if (info.name == name) {
                out.emplace(info);
                break;
            }

        return out;
    }

    std::string TensorRTImpl::Print() {
        std::string str;
        str += "[TensorRT] ********************** Exec by TensorRT ********************** ";
        str += "\n[TensorRT] Input count: ";
        str += std::to_string(inputs_info_.size());
        str += "\n[TensorRT] Input names: ";
        for (auto & info : inputs_info_)
            str += info.name + " ";
        str += "\n[TensorRT] Input shapes:";
        for (size_t i = 0; i < inputs_info_.size(); ++i) {
            const std::vector<int> &shape = inputs_info_[i].shape;
            for (auto j : shape)
                str += " " + std::to_string(j);
            if (i != inputs_info_.size() - 1)
                str += ",";
        }

        str += "\n[TensorRT] Output count: ";
        str += std::to_string(outputs_info_.size());
        str += "\n[TensorRT] Output names: ";
        for (auto & info : outputs_info_)
            str += info.name + " ";
        str += "\n[TensorRT] Output shapes:";
        for (size_t i = 0; i < outputs_info_.size(); ++i) {
            const std::vector<int> &shape = outputs_info_[i].shape;
            for (auto j : shape)
                str += " " + std::to_string(j);
            if (i != outputs_info_.size() - 1)
                str += ",";
        }

        return str;
    }

    std::uint64_t TensorRTImpl::GetRandomUInt64(std::uint64_t begin, std::uint64_t end) {
        std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
        std::int64_t seek = std::chrono::system_clock::to_time_t(tp);
        static std::default_random_engine random_eng(seek % 100);
        std::uniform_int_distribution<size_t> ud(begin, end);

        return ud(random_eng);
    }

    Error TensorRTImpl::CheckPath(const std::string &path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin.good())    return Error::ErrorInvalidModelFile;

        return Error::Success;
    }

    Error TensorRTImpl::CheckCudaDevice(
            const std::unordered_map<std::uint32_t, std::uint32_t> &cuda_device_concurrency_) {
        int device_count;
        cudaGetDeviceCount(&device_count);

        if (cuda_device_concurrency_.empty() || device_count <= 0)
            return Error::ErrorInferEnvError;

        for (auto &pair : cuda_device_concurrency_) {
            const std::uint32_t id = std::get<0>(pair);
            if (id > device_count - 1)      return Error::ErrorInferEnvError;
        }

        return Error::Success;
    }

    std::shared_ptr<TensorRT> TensorRT::MakeEngine(Error *ec, const Config& config) {
        Error err = TensorRTImpl::CheckPath(config.path);
        if (err) {
            *ec = err;

            return nullptr;
        }

        err = TensorRTImpl::CheckCudaDevice(config.GetCudaDeviceConcurrency());
        if (err) {
            *ec = err;

            return nullptr;
        }

        std::ifstream fin(config.path, std::ios::binary);
        fin.seekg(0, std::ifstream::end);
        std::int64_t filesize = fin.tellg();
        fin.seekg(0, std::ifstream::beg);
        std::vector<char> engine_data(filesize);
        fin.read(engine_data.data(), filesize);
        if (!fin.good()) {
            *ec = Error::ErrorInvalidModelFile;

            return nullptr;
        }
        fin.close();

        std::shared_ptr<TensorRTImpl> trt = std::make_shared<TensorRTImpl>();
        err = trt->Create(config.GetCudaDeviceConcurrency(), engine_data, config.name);
        if (err) {
            *ec = err;

            return nullptr;
        }
        *ec = Error::Success;

        return trt;
    }
}








