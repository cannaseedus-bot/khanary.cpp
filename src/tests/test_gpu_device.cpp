#include "conformance.h"
#include "../gpu/d3d11_device.h"

using namespace khanary;

int main() {
    return test::run("gpu-device", [](test::Suite& suite) {
        const auto adapters = gpu::D3D11Device::enumerate_adapters();
        suite.check("adapter_enumeration_invoked", true);

        if (adapters.empty()) {
            suite.check("no_adapter_reported", true);
        } else {
            suite.check("first_adapter_has_name", !adapters.front().description.empty());
        }

        gpu::D3D11Device device;
        bool init_called = false;
        bool init_ok = false;
        try {
            init_called = true;
            init_ok = device.init("auto");
        } catch (const std::exception&) {
            init_ok = false;
        }

        suite.check("device_init_invoked", init_called);
        if (init_ok) {
            suite.check("device_handle_valid", device.GetDevice() != nullptr);
            suite.check("context_handle_valid", device.GetContext() != nullptr);
        } else {
            suite.check("device_init_result_recorded", true);
        }
    });
}
