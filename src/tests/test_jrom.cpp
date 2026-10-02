#include "conformance.h"
#include "../io/jrom.h"

#include <filesystem>
#include <iostream>
#include <vector>

using namespace khanary;

static void require(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "[FAIL] " << msg << "\n";
        throw khanary::test::Failure(msg);
    }
}

static int acceptance_main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("khanary_jrom_epistemic_" + std::to_string(std::filesystem::file_time_type::clock::now().time_since_epoch().count()) + ".jsonl");

    {
        io::JROM jrom(path.string());
        require(jrom.is_open(), "jrom opened");
        require(io::JROM::is_valid_epistemic_type("plan"), "plan is a valid epistemic type");
        require(!io::JROM::is_valid_epistemic_type("authority_override"), "authority_override is not a valid epistemic type");

        const auto s1 = jrom.log("semantic_envelope", "{\"ok\":true}", "plan");
        const auto s2 = jrom.log("semantic_envelope", "{\"ok\":false}", "authority_override");
        const auto s3 = jrom.log("legacy_event", "{\"ok\":true}");
        const auto s4 = jrom.log("hyperbolic_event",
                                 "{\"key\":\"kg:advisor:yax:why\",\"origin\":\"turn:current\",\"semantic_authority\":false}",
                                 "observation");
        require(s1 == 1 && s2 == 2 && s3 == 3 && s4 == 4, "sequence increments");
    }

    {
        io::JROM replay(path.string());
        std::vector<io::JROMEvent> events = replay.read_all();
        require(events.size() == 4, "replay reads all events");
        require(events[0].type == "semantic_envelope", "event type preserved");
        require(events[0].epistemic_type == "plan", "valid epistemic type preserved");
        require(events[0].payload_json == "{\"ok\":true}", "payload preserved");
        require(events[1].epistemic_type == "generated_output", "invalid epistemic type normalized");
        require(events[2].epistemic_type.empty(), "legacy event omits epistemic type");
        require(events[3].type == "hyperbolic_event", "hyperbolic event type preserved");
        require(events[3].epistemic_type == "observation", "hyperbolic event epistemic type preserved");
    }

    std::filesystem::remove(path);
    std::cout << "[PASS] jrom_epistemic_type replay contract\n";
    return 0;
}

int main() {
    return khanary::test::legacy("jrom", [] { return acceptance_main(); });
}
