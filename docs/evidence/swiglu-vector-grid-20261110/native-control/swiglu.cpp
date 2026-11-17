// Replay the public FP32 short-row shapes through the installed native SwiGLU.
// Run under msprof to capture vector task times. Input/output files are raw FP32.
#include "acl/acl.h"
#include "aclnnop/aclnn_swi_glu.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int status, const char *operation) {
    if (status != 0) {
        std::cerr << operation << " failed (" << status << ") " << aclGetRecentErrMsg() << '\n';
        std::exit(1);
    }
}

struct Tensor {
    void *data{};
    aclTensor *desc{};
    size_t bytes{};
    explicit Tensor(const std::vector<int64_t> &shape) {
        std::vector<int64_t> strides(shape.size());
        int64_t count = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            strides[i] = count;
            count *= shape[i];
        }
        bytes = count * sizeof(float);
        check(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "tensor allocation");
        desc = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                               ACL_FORMAT_ND, shape.data(), shape.size(), data);
        if (!desc) throw std::runtime_error("aclCreateTensor failed");
    }
    ~Tensor() {
        aclDestroyTensor(desc);
        aclrtFree(data);
    }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
};

int main(int argc, char **argv) {
    if (argc != 4 || (std::string(argv[1]) != "11" && std::string(argv[1]) != "13")) {
        std::cerr << "usage: native_swiglu {11|13} input.f32 output.f32\n";
        return 2;
    }
    const bool padded = std::string(argv[1]) == "11";
    auto shape = padded ? std::vector<int64_t>{3,7,13,1018} : std::vector<int64_t>{11,13,16,67};
    const int64_t dim = padded ? -1 : 2;
    auto outshape = shape;
    outshape.at(dim < 0 ? shape.size() - 1 : size_t(dim)) /= 2;
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream{};
    check(aclrtCreateStream(&stream), "stream");
    {
        Tensor x(shape), y(outshape);
        std::vector<float> host(x.bytes/sizeof(float));
        std::ifstream input(argv[2], std::ios::binary);
        if (!input.read(reinterpret_cast<char *>(host.data()), x.bytes) || input.peek() != EOF)
            throw std::runtime_error("unexpected input size");
        check(aclrtMemcpy(x.data, x.bytes, host.data(), x.bytes, ACL_MEMCPY_HOST_TO_DEVICE), "input");
        void *workspace = nullptr;
        uint64_t capacity = 0;
        for (int i = 0; i < 17; ++i) {
            aclOpExecutor *executor = nullptr;
            uint64_t bytes = 0;
            check(aclnnSwiGluGetWorkspaceSize(x.desc, dim, y.desc, &bytes, &executor), "workspace query");
            if (bytes > capacity) {
                if (workspace) check(aclrtFree(workspace), "free workspace");
                check(aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "workspace allocation");
                capacity = bytes;
            }
            check(aclnnSwiGlu(workspace, bytes, executor, stream), "SwiGLU");
            check(aclrtSynchronizeStream(stream), "synchronize");
        }
        check(aclrtMemcpy(host.data(), y.bytes, y.data, y.bytes, ACL_MEMCPY_DEVICE_TO_HOST), "output");
        std::ofstream out(argv[3], std::ios::binary);
        if (!out.write(reinterpret_cast<char *>(host.data()), y.bytes).flush())
            throw std::runtime_error("output write failed");
        if (workspace) check(aclrtFree(workspace), "free workspace");
    }
    check(aclrtDestroyStream(stream), "destroy stream");
    check(aclrtResetDevice(0), "reset device");
    check(aclFinalize(), "aclFinalize");
}
