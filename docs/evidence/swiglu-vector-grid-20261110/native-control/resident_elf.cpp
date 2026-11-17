// Replay a two-pointer AscTile ELF with the allocation schedule of the native control.
#include "acl/acl.h"
#if __has_include("experiment/runtime/runtime/rt.h")
#include "experiment/runtime/runtime/rt.h"
#else
#include "runtime/rt.h"
#endif
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static void check(int rc, const char* op) {
    if (rc != 0) {
        std::cerr << op << " returned " << rc << '\n';
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string which = argv[1];
    if (which != "11" && which != "13") return 2;
    const char* name = std::getenv("REPLAY_ENTRY");
    if (!name || !*name) return 2;
    const size_t lanes = which == "11" ? 273 * 509 : 143 * 536;
    const uint32_t grid = which == "11" ? 35 : 36;
    const size_t bytes = lanes * sizeof(float);
    std::vector<float> host(2 * lanes);
    std::ifstream input(argv[2], std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(host.data()), 2 * bytes) || input.peek() != EOF) return 2;
    std::ifstream elf_file("device.o", std::ios::binary);
    std::vector<char> elf((std::istreambuf_iterator<char>(elf_file)), {});
    if (elf.size() < 4 || std::string(elf.data(), 4) != "\x7f" "ELF") return 2;

    check(aclInit(nullptr), "init");
    check(aclrtSetDevice(0), "device");
    aclrtStream stream;
    check(aclrtCreateStream(&stream), "stream");
    void *src, *dst;
    const std::string policy = std::getenv("REPLAY_PAGES") ? std::getenv("REPLAY_PAGES") : "normal";
    if (policy != "huge" && policy != "normal") return 2;
    const auto allocation = policy == "huge" ? ACL_MEM_MALLOC_HUGE_ONLY : ACL_MEM_MALLOC_NORMAL_ONLY;
    const std::string staging = std::getenv("REPLAY_COPIES") ? std::getenv("REPLAY_COPIES") : "once";
    // Bits select the transfers between calls while retaining both addresses.
    // Keep historical names for the original two schedules.
    unsigned transfers = 0;
    if (staging == "each") transfers = 7;
    else if (staging.size() == 1 && staging[0] >= '0' && staging[0] <= '7')
        transfers = static_cast<unsigned>(staging[0] - '0');
    else if (staging != "once") return 2;
    std::vector<float> output(lanes);
    check(aclrtMalloc(&src, 2 * bytes, allocation), "allocate input");
    check(aclrtMalloc(&dst, bytes, allocation), "allocate output");
    check(aclrtMemcpy(src, 2 * bytes, host.data(), 2 * bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");

    rtDevBinary_t device{};
    device.magic = 0x41415246U; // RT_DEV_BINARY_MAGIC_ELF_AIVEC
    device.data = reinterpret_cast<decltype(device.data)>(elf.data());
    device.length = elf.size();
    void* binary = nullptr;
    check(rtDevBinaryRegister(&device, &binary), "register binary");
    check(rtFunctionRegister(binary, name, name, name, 0), "register function");
    uintptr_t args[] = {reinterpret_cast<uintptr_t>(src), reinterpret_cast<uintptr_t>(dst)};
    for (int i = 0; i < 17; ++i) {
        if (transfers & 1U)
            check(aclrtMemcpy(src, 2 * bytes, host.data(), 2 * bytes, ACL_MEMCPY_HOST_TO_DEVICE), "stage input");
        if (transfers & 2U)
            check(aclrtMemcpy(dst, bytes, output.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "stage output");
        check(rtKernelLaunch(name, grid, args, sizeof(args), nullptr, stream), "launch");
        check(aclrtSynchronizeStream(stream), "synchronize");
        if (transfers & 4U)
            check(aclrtMemcpy(output.data(), bytes, dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read output");
    }
    check(aclrtMemcpy(host.data(), bytes, dst, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
    std::ofstream out(argv[3], std::ios::binary);
    if (!out.write(reinterpret_cast<char*>(host.data()), bytes).flush()) return 1;
    check(rtDevBinaryUnRegister(binary), "unregister binary");
    check(aclrtFree(dst), "free output");
    check(aclrtFree(src), "free input");
    check(aclrtDestroyStream(stream), "destroy stream");
    check(aclrtResetDevice(0), "reset device");
    check(aclFinalize(), "finalize");
}
