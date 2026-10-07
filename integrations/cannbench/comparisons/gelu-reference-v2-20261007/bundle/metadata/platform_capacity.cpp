// Host metadata C ABI; no device initialization, launch, or resource setters.
#include <cstdint>
#include <string>
#include "platform/platform_info.h"
#include "platform/platform_ascendc.h"

extern "C" int gelu_platform_ub(const char *soc, uint64_t *ub, uint32_t *cann_status) noexcept {
    if (ub == nullptr || cann_status == nullptr) return 1;
    *ub = 0;
    *cann_status = 0;
    if (soc == nullptr) return 1;
    size_t length = 0;
    for (; length <= 96 && soc[length] != '\0'; ++length) {
        const char ch = soc[length];
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
              (ch >= '0' && ch <= '9') || ch == '_')) return 1;
    }
    if (length == 0 || length > 96) return 1;
    try {
        auto &manager = fe::PlatformInfoManager::Instance();
        *cann_status = manager.InitializePlatformInfo();
        if (*cann_status != 0) return 2;
        fe::PlatFormInfos info;
        fe::OptionalInfos optional;
        *cann_status = manager.GetPlatformInfos(std::string(soc, length), info, optional);
        if (*cann_status != 0) return 3;
        platform_ascendc::PlatformAscendC platform(&info);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ub);
        return *ub > 0 ? 0 : 4;
    } catch (...) {
        *ub = 0;
        return 5;
    }
}
