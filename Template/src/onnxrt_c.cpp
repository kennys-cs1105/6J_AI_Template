//
// Created by zyh on 12/1/22.
//
#include <fstream>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <onnxruntime_c_api.h>
#include <random>

#include "onnxrt_c.h"

namespace nn {
    OnnxRTImpl_C::OnnxRAII::~OnnxRAII() {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (status_)     api->ReleaseStatus(status_);
        if (session_)  api->ReleaseSession(session_);
        if (memory_info_)    api->ReleaseMemoryInfo(memory_info_);
        if (type_info_)  api->ReleaseTypeInfo(type_info_);
        if (allocator_) api->ReleaseAllocator(allocator_);
    }

    OrtSession *OnnxRTImpl_C::OnnxRAII::GetSession() {
        return session_;
    }

    OrtSession **OnnxRTImpl_C::OnnxRAII::GetSessionMutablePtr() {
        return &session_;
    }

    OrtMemoryInfo *OnnxRTImpl_C::OnnxRAII::GetMemoryInfo() {
        return memory_info_;
    }

    OrtStatus *OnnxRTImpl_C::OnnxRAII::GetStatus() {
        return status_;
    }

    OrtTypeInfo *OnnxRTImpl_C::OnnxRAII::GetTypeInfo() {
        return type_info_;
    }

    OrtAllocator *OnnxRTImpl_C::OnnxRAII::GetAllocator() {
        return allocator_;
    }

    void OnnxRTImpl_C::OnnxRAII::FreeAndSetOrt(OrtStatus *status) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (status_)
            api->ReleaseStatus(status_);
        status_ = status;
    }

    void OnnxRTImpl_C::OnnxRAII::FreeAndSetOrt(OrtTypeInfo **type_info) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (type_info_)
            api->ReleaseTypeInfo(type_info_);
        type_info_ = *type_info;
        *type_info = nullptr;
    }

    void OnnxRTImpl_C::OnnxRAII::SetOrt(OrtSession** session) {
        session_ = *session;
        *session = nullptr;
    }

    void OnnxRTImpl_C::OnnxRAII::SetOrt(OrtMemoryInfo **memory_info) {
        memory_info_ = *memory_info;
        *memory_info = nullptr;
    }

    void OnnxRTImpl_C::OnnxRAII::SetOrt(OrtAllocator **allocator) {
        allocator_ = *allocator;
        *allocator = nullptr;
    }

    OnnxRTImpl_C::~OnnxRTImpl_C() noexcept {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (options_)
            api->ReleaseSessionOptions(options_);
        if (env_)
            api->ReleaseEnv(env_);
    }

    OnnxRTImpl_C::OnnxRTImpl_C(OnnxRTImpl_C &&impl) noexcept :
        inputs_info_(std::move(impl.inputs_info_)),
        outputs_info_(std::move(impl.outputs_info_)),
        executors_(std::move(impl.executors_)) {
        env_ = impl.env_;
        impl.env_ = nullptr;

        options_ = impl.options_;
        impl.options_ = nullptr;
    }

    OnnxRTImpl_C &OnnxRTImpl_C::operator=(OnnxRTImpl_C &&impl) noexcept {
        inputs_info_ = std::move(impl.inputs_info_);
        outputs_info_ = std::move(impl.outputs_info_);
        executors_ = std::move(impl.executors_);

        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (options_)
            api->ReleaseSessionOptions(options_);
        if (env_)
            api->ReleaseEnv(env_);

        env_ = impl.env_;
        impl.env_ = nullptr;

        options_ = impl.options_;
        impl.options_ = nullptr;

        return *this;
    }

    Error OnnxRTImpl_C::CheckPath(const std::string &path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin.good())
            return Error::ErrorInvalidModelFile;

        return Error::Success;
    }

    std::uint32_t OnnxRTImpl_C::CheckThreads(std::uint32_t threads) {
        std::uint32_t max_t = std::thread::hardware_concurrency();

        return std::clamp(threads, 0u, max_t);
    }

    std::uint32_t OnnxRTImpl_C::CheckConcurrency(std::uint32_t concurrency) {
        return std::clamp(concurrency, 1u, UINT32_MAX);
    }

    std::uint64_t OnnxRTImpl_C::GetRandomUInt64(std::uint64_t begin, std::uint64_t end) {
        std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
        std::int64_t seek = std::chrono::system_clock::to_time_t(tp);
        static std::default_random_engine random_eng(seek % 100);
        std::uniform_int_distribution<size_t> ud(begin, end);

        return ud(random_eng);
    }

    const std::vector<OnnxRTImpl_C::TensorInfo> &OnnxRTImpl_C::GetInputInfo() {
        return inputs_info_;
    }

    const std::vector<OnnxRTImpl_C::TensorInfo> &OnnxRTImpl_C::GetOutputInfo() {
        return outputs_info_;
    }

    std::optional<OnnxRTImpl_C::TensorInfo> OnnxRTImpl_C::GetInputInfoByName(const std::string &name) {
        std::optional<OnnxRTImpl_C::TensorInfo> in;
        for (auto &info: inputs_info_)
            if (info.name == name) {
                in.emplace(info);
                break;
            }

        return in;
    }

    std::optional<OnnxRTImpl_C::TensorInfo> OnnxRTImpl_C::GetOutputInfoByName(const std::string &name) {
        std::optional<OnnxRTImpl_C::TensorInfo> out;
        for (auto &info: outputs_info_)
            if (info.name == name) {
                out.emplace(info);
                break;
            }

        return out;
    }

    std::string OnnxRTImpl_C::Print() {
        std::string str;
        str += "[OnnxRT] ********************** Exec by OnnxRT ********************** ";
        str += "\n[OnnxRT] Input count " + std::to_string(inputs_info_.size());
        str += "\n[OnnxRT] Input names: ";
        for (const auto & i : inputs_info_)
            str += i.name + " ";
        str += "\n[OnnxRT] Input shapes:";
        for (size_t i = 0; i < inputs_info_.size(); ++i) {
            const std::vector<std::int64_t>& shape = inputs_info_[i].shape;
            for (std::int64_t j : shape)
                str += " " + std::to_string(j);
            if (i != inputs_info_.size() - 1)
                str += ",";
        }

        str += "\n[OnnxRT] Output count " + std::to_string(outputs_info_.size());
        str += "\n[OnnxRT] Output names: ";
        for (const auto & i : outputs_info_)
            str += i.name + " ";
        str += "\n[OnnxRT] Output shapes:";
        for (size_t i = 0; i < outputs_info_.size(); ++i) {
            const std::vector<std::int64_t>& shape = outputs_info_[i].shape;
            for (std::int64_t j : shape)
                str += " " + std::to_string(j);
            if (i != outputs_info_.size() - 1)
                str += ",";
        }

        return str;
    }

    Error OnnxRTImpl_C::SetTensorInfo(OrtSession *sess) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OnnxRAII raii;

        OrtMemoryInfo *memory_info = nullptr;
        raii.FreeAndSetOrt(api->CreateCpuMemoryInfo(
                OrtAllocatorType::Invalid, OrtMemType::OrtMemTypeDefault, &memory_info));
        raii.SetOrt(&memory_info);
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        OrtAllocator *allocator = nullptr;
        raii.FreeAndSetOrt(api->CreateAllocator(
                sess, raii.GetMemoryInfo(), &allocator));
        raii.SetOrt(&allocator);
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        size_t count = 0;
        raii.FreeAndSetOrt(api->SessionGetInputCount(sess, &count));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        for (size_t i = 0; i < count; ++i) {
            OrtTypeInfo *type_info = nullptr;
            raii.FreeAndSetOrt(api->SessionGetInputTypeInfo(sess, i, &type_info));
            raii.FreeAndSetOrt(&type_info);
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            const OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
            raii.FreeAndSetOrt(api->CastTypeInfoToTensorInfo(
                    raii.GetTypeInfo(), &type_shape_info));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            size_t dim_count = 0;
            raii.FreeAndSetOrt(api->GetDimensionsCount(type_shape_info, &dim_count));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            std::vector<std::int64_t> shape64(dim_count);
            raii.FreeAndSetOrt(api->GetDimensions(type_shape_info, shape64.data(), dim_count));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            char *name = nullptr;
            raii.FreeAndSetOrt(api->SessionGetInputName(sess, i, raii.GetAllocator(), &name));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            inputs_info_.emplace_back(name, std::move(shape64));
        }

        raii.FreeAndSetOrt(api->SessionGetOutputCount(sess, &count));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        for (size_t i = 0; i < count; ++i) {
            OrtTypeInfo *type_info = nullptr;
            raii.FreeAndSetOrt(api->SessionGetOutputTypeInfo(sess, i, &type_info));
            raii.FreeAndSetOrt(&type_info);
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            const OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
            raii.FreeAndSetOrt(api->CastTypeInfoToTensorInfo(
                    raii.GetTypeInfo(), const_cast<const OrtTensorTypeAndShapeInfo **>(&type_shape_info)));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            size_t dim_count = 0;
            raii.FreeAndSetOrt(api->GetDimensionsCount(type_shape_info, &dim_count));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            std::vector<std::int64_t> shape64(dim_count);
            raii.FreeAndSetOrt(api->GetDimensions(type_shape_info, shape64.data(), dim_count));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            char *name = nullptr;
            raii.FreeAndSetOrt(api->SessionGetOutputName(sess, i, raii.GetAllocator(), &name));
            if (raii.GetStatus())    return Error::ErrorInferEnvError;

            outputs_info_.emplace_back(name, std::move(shape64));
        }

        return Error::Success;
    }

    Error OnnxRTImpl_C::Create(const std::string &path, const std::string &name,
                             std::uint32_t intra, std::uint32_t inter, std::uint32_t concurrency) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OnnxRAII raii;

        raii.FreeAndSetOrt(api->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, name.c_str(), &env_));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        raii.FreeAndSetOrt(api->CreateSessionOptions(&options_));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        raii.FreeAndSetOrt(api->SetSessionGraphOptimizationLevel(options_, GraphOptimizationLevel::ORT_ENABLE_EXTENDED));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        raii.FreeAndSetOrt(api->SetInterOpNumThreads(options_, static_cast<int>(inter)));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        raii.FreeAndSetOrt(api->SetIntraOpNumThreads(options_, static_cast<int>(intra)));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        for (size_t i = 0; i < concurrency; ++i) {
            OrtSession* sess = nullptr; // sess: nullptr
            raii.FreeAndSetOrt(api->CreateSession(env_, path.c_str(), options_, &sess));
            raii.SetOrt(&sess);      // sess: RAII
            if (raii.GetStatus())    return Error::ErrorInvalidModelFile;

            if (GetInputInfo().empty() && GetOutputInfo().empty()) {
                Error err = SetTensorInfo(raii.GetSession());
                if (err)    return err;
            }

            OnnxExecutor exe;
            Error err = exe.Create(raii.GetSessionMutablePtr(), inputs_info_, outputs_info_);

            if (err)    return err;

            executors_.emplace_back(std::move(exe));
        }

        return Error::Success;
    }

    Error OnnxRTImpl_C::Run(const float **inputs, const std::vector<std::string> &input_names, float **outputs,
                            const std::vector<std::string> &output_names) {
        if (input_names.size() != inputs_info_.size())
            return Error::ErrorInvalidTensor;
        if (output_names.size() != outputs_info_.size())
            return Error::ErrorInvalidTensor;

        std::uint64_t session_id = GetRandomUInt64(0, executors_.size() - 1);

        Error ec = executors_[session_id].Run(inputs, input_names, outputs, output_names, *this);
        if (ec)     return ec;

        return Error::Success;
    }

    OnnxRTImpl_C::OnnxExecutor::OnnxExecutor(OnnxExecutor&& exe) noexcept :
        i_values_(std::move(exe.i_values_)), o_values_(std::move(exe.o_values_)) {
        session_ = exe.session_;
        exe.session_ = nullptr;

        allocator_ = exe.allocator_;
        exe.allocator_ = nullptr;
    }

    OnnxRTImpl_C::OnnxExecutor &OnnxRTImpl_C::OnnxExecutor::operator=(OnnxRTImpl_C::OnnxExecutor &&exe) noexcept {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        for (auto v : i_values_)
            if (v)  api->ReleaseValue(v);
        for (auto v : o_values_)
            if (v)  api->ReleaseValue(v);
        if (allocator_) api->ReleaseAllocator(allocator_);
        if (session_)   api->ReleaseSession(session_);

        i_values_ = std::move(exe.i_values_);
        o_values_ = std::move(exe.o_values_);

        session_ = exe.session_;
        exe.session_ = nullptr;

        allocator_ = exe.allocator_;
        exe.allocator_ = nullptr;

        return *this;
    }

    OnnxRTImpl_C::OnnxExecutor::~OnnxExecutor() {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        for (auto v : i_values_)
            if (v)  api->ReleaseValue(v);
        for (auto v : o_values_)
            if (v)  api->ReleaseValue(v);
        if (allocator_) api->ReleaseAllocator(allocator_);
        if (session_)   api->ReleaseSession(session_);
    }

    Error OnnxRTImpl_C::OnnxExecutor::Create(OrtSession **sess,
                   const std::vector<TensorInfo>& inputs_info_, const std::vector<TensorInfo>& outputs_info_) {
        session_ = *sess;
        *sess = nullptr;

        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OnnxRAII raii;

        OrtMemoryInfo *memory_info = nullptr;
        raii.FreeAndSetOrt(api->CreateCpuMemoryInfo(
                OrtAllocatorType::Invalid, OrtMemType::OrtMemTypeDefault, &memory_info));
        raii.SetOrt(&memory_info);
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        raii.FreeAndSetOrt(api->CreateAllocator(
                session_, raii.GetMemoryInfo(), &allocator_));
        if (raii.GetStatus())    return Error::ErrorInferEnvError;

        size_t count = inputs_info_.size();
        i_values_.assign(count, nullptr);

        for (size_t i = 0; i < count; ++i) {
            raii.FreeAndSetOrt(api->CreateTensorAsOrtValue(allocator_, inputs_info_[i].shape.data(),
                                                           inputs_info_[i].shape.size(),
                                                           ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           i_values_.data() + i));
            if (raii.GetStatus())   return Error::ErrorInferEnvError;
        }

        count = outputs_info_.size();
        o_values_.assign(count, nullptr);

        for (size_t i = 0; i < count; ++i) {
            raii.FreeAndSetOrt(api->CreateTensorAsOrtValue(allocator_, outputs_info_[i].shape.data(),
                                        outputs_info_[i].shape.size(),
                                        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                        o_values_.data() + i));
            if (raii.GetStatus())   return Error::ErrorInferEnvError;
        }

        return Error::Success;
    }

    Error OnnxRTImpl_C::OnnxExecutor::Run(const float **inputs, const std::vector<std::string> &input_names, float **outputs,
                                    const std::vector<std::string> &output_names, const OnnxRTImpl_C& onnx) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OnnxRAII raii;

        size_t input_size = onnx.inputs_info_.size();

        std::vector<size_t> input_counts(input_size, 0);
        std::vector<size_t> input_ids(input_size, 0);

        for (size_t i = 0; i < input_size; ++i) {
            const auto& info = onnx.inputs_info_[i];
            const std::string& binding_name = info.name;
            auto iter = std::find(input_names.begin(), input_names.end(), binding_name);
            if (iter == input_names.end())
                return Error::ErrorInvalidTensor;
            size_t idx = iter - input_names.begin();
            std::int64_t count = std::accumulate(info.shape.begin(), info.shape.end(), 1, std::multiplies<>());

            input_counts[i] = count;
            input_ids[i] = idx;
        }

        std::vector<const char*> inames;
        for (auto& info: onnx.inputs_info_)
            inames.push_back(info.name.c_str());

        size_t output_size = onnx.outputs_info_.size();

        std::vector<size_t> output_counts(output_size, 0);
        std::vector<size_t> output_ids(output_size, 0);

        for (size_t i = 0; i < output_size; ++i) {
            const auto& info = onnx.outputs_info_[i];
            const std::string& binding_name = info.name;
            auto iter = std::find(output_names.begin(), output_names.end(), binding_name);
            if (iter == output_names.end())     return Error::ErrorInvalidTensor;
            size_t idx = iter - output_names.begin();
            std::int64_t count = std::accumulate(info.shape.begin(), info.shape.end(), 1, std::multiplies<>());

            output_counts[i] = count;
            output_ids[i] = idx;
        }

        std::vector<const char*> onames;
        for (auto& info: onnx.outputs_info_)
            onames.push_back(info.name.c_str());

        {
            std::lock_guard<std::mutex> lg(mutex_);
            for (size_t i = 0; i < input_size; ++i) {
                void* ibuffer = nullptr;
                raii.FreeAndSetOrt(api->GetTensorMutableData(i_values_[i], &ibuffer));
                if (raii.GetStatus())    return Error::ErrorInvalidTensor;
                memcpy(ibuffer, inputs[input_ids[i]], input_counts[i] * sizeof(float));
            }

            raii.FreeAndSetOrt(api->Run(session_, nullptr,
                                        inames.data(), i_values_.data(), input_size,
                                        onames.data(), output_size, o_values_.data()));
            if (raii.GetStatus())    return Error::ErrorInferenceError;

            for (size_t i = 0; i < output_size; ++i) {
                void* obuffer = nullptr;
                raii.FreeAndSetOrt(api->GetTensorMutableData(o_values_[i], &obuffer));
                if (raii.GetStatus())    return Error::ErrorInvalidTensor;
                memcpy(outputs[output_ids[i]], obuffer, output_counts[i] * sizeof(float));
            }
        }

        return Error::Success;
    }

    std::shared_ptr<OnnxRT> OnnxRT::MakeEngine(Error *ec, const Config &config) {
        Error err = OnnxRTImpl_C::CheckPath(config.path);
        if (err) {
            *ec = err;

            return nullptr;
        }

        std::uint32_t intra = OnnxRTImpl_C::CheckThreads(config.intra_op_threads);
        std::uint32_t inter = OnnxRTImpl_C::CheckThreads(config.inter_op_threads);
        std::uint32_t concurrency = OnnxRTImpl_C::CheckConcurrency(config.concurrency);

        std::shared_ptr<OnnxRTImpl_C> onnx = std::make_shared<OnnxRTImpl_C>();
        err = onnx->Create(config.path, config.name, intra, inter, concurrency);
        if (err) {
            *ec = err;

            return nullptr;
        }
        *ec = Error::Success;

        return onnx;
    }
}





















