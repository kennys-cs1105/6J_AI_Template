//
// Created by zyh on 12/1/22.
//

#ifndef AI_TEMPLATE_ONNXRT_C_H
#define AI_TEMPLATE_ONNXRT_C_H

#include <thread>
#include <mutex>

#include "../include/inference.h"

namespace nn {
    class OnnxRTImpl_C final : public OnnxRT {
    public:
        // Ctor
        OnnxRTImpl_C() : env_(nullptr), options_(nullptr) {}

        // Dtor
        ~OnnxRTImpl_C() noexcept final;

        // Move ctor
        OnnxRTImpl_C(OnnxRTImpl_C &&) noexcept;

        // Move assignment operator
        OnnxRTImpl_C &operator=(OnnxRTImpl_C&&) noexcept;

        // Copy ctor
        OnnxRTImpl_C(const OnnxRTImpl_C&) = delete;

        // Copy assignment operator
        OnnxRTImpl_C &operator=(const OnnxRTImpl_C &) = delete;

        Error SetTensorInfo(OrtSession* sess);

        Error Create(const std::string& path, const std::string& name,
                     std::uint32_t intra, std::uint32_t inter, std::uint32_t concurrency);

        Error Run(const float** inputs, const std::vector<std::string> &input_names,
                  float** outputs, const std::vector<std::string> &output_names) final;

        std::string Print() final;

        const std::vector<TensorInfo>& GetInputInfo() final;

        const std::vector<TensorInfo>& GetOutputInfo() final;

        std::optional<TensorInfo> GetInputInfoByName(const std::string &name) final;

        std::optional<TensorInfo> GetOutputInfoByName(const std::string &name) final;

        static Error CheckPath(const std::string& path);

        static std::uint32_t CheckThreads(std::uint32_t threads);

        static std::uint32_t CheckConcurrency(std::uint32_t concurrency);

        static std::uint64_t GetRandomUInt64(std::uint64_t begin, std::uint64_t end);

    private:
        class OnnxExecutor {
        private:
            OrtSession* session_;
            OrtAllocator* allocator_;

            std::vector<OrtValue*> i_values_;
            std::vector<OrtValue*> o_values_;
            std::mutex mutex_;
        public:
            OnnxExecutor() : session_(nullptr), allocator_(nullptr) {}
            ~OnnxExecutor() noexcept;
            OnnxExecutor(OnnxExecutor&&) noexcept;
            OnnxExecutor &operator=(OnnxExecutor&&) noexcept;

            OnnxExecutor(const OnnxExecutor&) = delete;
            OnnxExecutor &operator=(const OnnxExecutor &) = delete;

            Error Create(OrtSession** sess,
                         const std::vector<TensorInfo>& inputs_info_,
                         const std::vector<TensorInfo>& outputs_info_);

            Error Run(const float** inputs, const std::vector<std::string> &input_names,
                      float** outputs, const std::vector<std::string> &output_names,
                      const OnnxRTImpl_C& onnx);
        };

        class OnnxRAII {
        public:
            OnnxRAII() : status_(nullptr), memory_info_(nullptr),
                         type_info_(nullptr), session_(nullptr), allocator_(nullptr) {}
            ~OnnxRAII();

            OnnxRAII(OnnxRAII &&) = delete;
            OnnxRAII &operator=(OnnxRAII&&) = delete;
            OnnxRAII(const OnnxRAII&) = delete;
            OnnxRAII &operator=(const OnnxRAII &) = delete;

            void FreeAndSetOrt(OrtStatus*);
            void FreeAndSetOrt(OrtTypeInfo**);
            void SetOrt(OrtAllocator**);
            void SetOrt(OrtMemoryInfo**);
            void SetOrt(OrtSession**);

            OrtStatus* GetStatus();
            OrtMemoryInfo* GetMemoryInfo();
            OrtTypeInfo* GetTypeInfo();
            OrtAllocator* GetAllocator();
            OrtSession* GetSession();
            OrtSession** GetSessionMutablePtr();
        private:
            OrtStatus* status_;
            OrtMemoryInfo* memory_info_;
            OrtTypeInfo* type_info_;
            OrtSession* session_;
            OrtAllocator* allocator_;
        };

        std::vector<TensorInfo> inputs_info_;
        std::vector<TensorInfo> outputs_info_;
        std::vector<OnnxExecutor> executors_;
        OrtEnv* env_;
        OrtSessionOptions* options_;
    };
}

#endif //AI_TEMPLATE_ONNXRT_C_H
