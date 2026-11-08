#include "../src/llama-moe-cache-policy.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

using json = nlohmann::json;

struct step {
    int32_t layer;
    uint64_t graph;
    uint64_t clock;
    uint64_t latency_clock;
    bool force_resident;
    bool device;
    bool accessible;
    bool scored;
    size_t hits;
    size_t admissions;
    size_t temporary;
    std::vector<int32_t> ids;
};

static json replay(const std::vector<step> & steps, int32_t layers, int32_t experts, int32_t slots, bool baseline) {
    moe_cache_policy policy;
    policy.init(layers, experts, slots);
    std::vector<int32_t> map(size_t(layers)*experts, -1);
    for (int32_t il = 0; il < layers; ++il) {
        policy.slot_map[il] = map.data() + size_t(il)*experts;
    }
    std::vector<moe_cache_policy::fill> fills;
    std::map<uint64_t, size_t> misses_by_graph;
    for (const auto & s : steps) {
        bool accessible = s.accessible;
        if (s.device && !baseline) {
            accessible = std::any_of(s.ids.begin(), s.ids.end(), [&](int32_t e) {
                return policy.slot_map[s.layer][e] < 0;
            });
        }
        size_t hits;
        if (!policy.plan(s.layer, s.ids.data(), s.ids.size(), fills, hits,
                         s.clock, s.force_resident, s.latency_clock, accessible)) {
            throw std::runtime_error("Too few slots for a captured route");
        }
        size_t temporary = policy.temporary.size();
        if (s.device && !accessible) {
            // The service observes completed device hits without looking at stale CPU slot maps.
            hits = policy.uniq.size();
            temporary = 0;
        }
        if (baseline && (hits != s.hits || fills.size() != s.admissions || temporary != s.temporary)) {
            throw std::runtime_error("Baseline replay mismatch in graph " + std::to_string(s.graph) +
                                     ", layer " + std::to_string(s.layer));
        }
        if (s.scored) {
            misses_by_graph[s.graph] += fills.size() + temporary;
        }
    }
    json result = {{"slots", slots}, {"verified_steps", baseline ? steps.size() : 0}, {"misses_by_graph", json::array()}};
    for (const auto & [graph, misses] : misses_by_graph) {
        result["misses_by_graph"].push_back({graph, misses});
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: test-moe-cache-replay REPLAY-INPUT.json\n";
        return 1;
    }
    try {
        std::ifstream file(argv[1]);
        const auto input = json::parse(file);
        json output = json::array();
        for (const auto & g : input.at("groups")) {
            const auto layers = g.at("layers").get<std::vector<int32_t>>();
            const int32_t experts = g.at("experts");
            if (layers.empty() || experts < 1 || experts > 32768 || layers.size() > size_t(INT32_MAX)/experts) {
                throw std::runtime_error("Invalid group layout");
            }
            const int32_t base = g.at("slots");
            const int32_t capacity = int32_t(layers.size())*experts;
            std::vector<step> steps;
            for (const auto & r : g.at("steps")) {
                const int32_t il = r.at("layer_id");
                const auto layer = std::find(layers.begin(), layers.end(), il);
                if (layer == layers.end()) {
                    throw std::runtime_error("Route references a layer outside its group");
                }
                const bool device = r.value("device_lookup", false);
                step s{int32_t(layer - layers.begin()), r.at("graph"), r.at("clock"), r.at("latency_clock"),
                       r.at("force_resident"), device, device ? r.at("device_miss").get<bool>() : r.at("local_cache").get<bool>(),
                       r.at("scored"), r.at("resident_hits"), r.at("admissions"), r.at("temporary"), {}};
                for (const auto & pair : r.at("route_counts")) {
                    const int32_t e = pair.at(0);
                    const int32_t count = pair.at(1);
                    if (e < 0 || e >= experts || count < 1 || size_t(count) + s.ids.size() > 1024*1024) {
                        throw std::runtime_error("Invalid routing histogram");
                    }
                    s.ids.insert(s.ids.end(), count, e);
                }
                if (s.ids.empty() || (!steps.empty() &&
                        (s.clock < steps.back().clock || s.latency_clock < steps.back().latency_clock))) {
                    throw std::runtime_error("Invalid route or planner clock");
                }
                steps.push_back(std::move(s));
            }
            json curves = json::array();
            for (const int32_t slots : g.at("tries")) {
                if (slots < 1 || slots > capacity) {
                    throw std::runtime_error("Invalid slot count");
                }
                curves.push_back(replay(steps, int32_t(layers.size()), experts, slots, slots == base));
            }
            output.push_back({{"group", g.at("cache_group")}, {"curves", std::move(curves)}});
        }
        std::cout << output.dump() << '\n';
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
