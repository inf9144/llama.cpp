#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

static constexpr int64_t moe_cache_max_batch = 32;
static constexpr int32_t moe_cache_stage_slots = 32;

static inline bool moe_cache_fits(int64_t tokens, int64_t used, int64_t experts, int64_t slots) {
    return tokens > 0 && used > 0 && used <= experts && slots >= used &&
        (slots >= experts || tokens <= slots / used);
}

// Advance with sequence progress, not wall time or the number of speculative calls.
// Revisiting rejected positions / restoring a checkpoint does not age the cache again.
// Starting a new prompt at zero starts a new segment without clearing useful history.
struct moe_cache_clock {
    uint64_t tokens = 0;
    std::unordered_map<int32_t, int64_t> high;

    void begin(int32_t seq, int64_t first, int64_t last) {
        auto it = high.emplace(seq, -1).first;
        if (first == 0 && it->second > 0) {
            it->second = -1;
        }
        if (last > it->second) {
            tokens += uint64_t(last - std::max(it->second, first - 1));
            it->second = last;
        }
    }
};

// One policy for decode, verify and prefill. Banks/budgets are unchanged.
// Scores count repeated weight requests, once per (layer, expert, progress stamp),
// not token assignments. The recent 1024 tokens have 4x the weight of older needs.
// 16 capped counters cover 4096 forward token positions. Saturation and the stamp
// prevent retries of rejected speculative tokens from gaining unlimited hotness.
struct moe_cache_policy {
    static constexpr uint64_t slice_tokens = 256;
    struct demand {
        uint64_t history = 0;
        uint64_t slice = 0;
        uint64_t stamp = UINT64_MAX;
        uint32_t frequency = 0; // secondary tie-break only
    };
    struct fill { int32_t expert; int32_t slot; };

    int32_t n_expert = 0;
    int32_t n_slots = 0;
    std::vector<int32_t *> slot_map;
    std::vector<int32_t> key_of;
    std::vector<demand> needs;
    std::vector<int32_t> uniq;
    std::vector<int32_t> temporary;
    std::vector<uint32_t> counts;
    std::vector<uint32_t> scores;

    void init(int32_t n_layer, int32_t experts, int32_t slots) {
        GGML_ASSERT(experts > 0 && slots > 0);
        n_expert = experts; n_slots = slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(slots, -1);
        needs.resize(size_t(n_layer)*experts);
        counts.resize(experts);
        scores.resize(needs.size());
    }

    uint32_t score(int32_t key, uint64_t slice) {
        demand & d = needs[key];
        const uint64_t delta = slice - d.slice;
        d.history = delta >= 16 ? 0 : d.history << (4*delta);
        d.slice = slice;
        const uint64_t pairs = (d.history & UINT64_C(0x0f0f0f0f0f0f0f0f)) +
            ((d.history >> 4) & UINT64_C(0x0f0f0f0f0f0f0f0f));
        const uint32_t total = (pairs*UINT64_C(0x0101010101010101)) >> 56;
        const uint32_t recent = (d.history & 15) + ((d.history >> 4) & 15) +
            ((d.history >> 8) & 15) + ((d.history >> 12) & 15);
        return total + 3*recent;
    }

    // force_resident is the latency-oriented all-resident execution plan for
    // small batches / legacy backends. Streaming permits bypass instead of churn.
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills,
              size_t & n_hit, uint64_t clock, bool force_resident) {
        fills.clear(); temporary.clear(); uniq.clear(); n_hit = 0;
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (counts[ids[i]]++ == 0) uniq.push_back(ids[i]);
        }
        if (force_resident && uniq.size() > size_t(n_slots)) return false;
        const uint64_t slice = clock / slice_tokens;
        for (int32_t e : uniq) {
            const int32_t key = il*n_expert + e;
            score(key, slice);
            demand & d = needs[key];
            if (d.stamp != clock) {
                if ((d.history & 15) != 15) ++d.history;
                d.frequency = (7*d.frequency + std::min(counts[e], uint32_t(65535))) / 8;
                d.stamp = clock;
            }
            scores[key] = score(key, slice);
            n_hit += slot_map[il][e] >= 0;
        }
        if (n_hit == uniq.size()) {
            std::sort(uniq.begin(), uniq.end());
            return true;
        }
        // All hits of this request stay pinned until the last matrix has read them.
        std::vector<int32_t> victims;
        victims.reserve(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            const int32_t key = key_of[s];
            if (key >= 0 && key/n_expert == il && counts[key % n_expert]) continue;
            if (key >= 0) scores[key] = score(key, slice);
            victims.push_back(s);
        }
        const auto victim_less = [&](int32_t a, int32_t b) {
            const int32_t ka = key_of[a], kb = key_of[b];
            if ((ka < 0) != (kb < 0)) return ka < 0;
            if (ka < 0) return a < b;
            if (scores[ka] != scores[kb]) return scores[ka] < scores[kb];
            if (needs[ka].stamp != needs[kb].stamp) return needs[ka].stamp < needs[kb].stamp;
            if (needs[ka].frequency != needs[kb].frequency) return needs[ka].frequency < needs[kb].frequency;
            return a < b;
        };
        const auto heap_less = [&](int32_t a, int32_t b) { return victim_less(b, a); };
        std::make_heap(victims.begin(), victims.end(), heap_less);
        std::vector<int32_t> misses;
        for (int32_t e : uniq) if (slot_map[il][e] < 0) misses.push_back(e);
        std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
            const int32_t ka = il*n_expert + a, kb = il*n_expert + b;
            if (scores[ka] != scores[kb]) return scores[ka] > scores[kb];
            if (needs[ka].frequency != needs[kb].frequency) return needs[ka].frequency > needs[kb].frequency;
            return a < b;
        });
        for (int32_t e : misses) {
            const int32_t candidate = il*n_expert + e;
            const int32_t s = victims.empty() ? -1 : victims.front();
            const int32_t old = s < 0 ? -1 : key_of[s];
            // Demand hysteresis: a one-off prefill scan cannot flush equally hot
            // residents. Empty slots are filled immediately, directly on upload.
            if (s < 0 || (!force_resident && old >= 0 &&
                    scores[candidate] <= scores[old] + std::max(1u, scores[old]/8))) {
                temporary.push_back(e);
                continue;
            }
            std::pop_heap(victims.begin(), victims.end(), heap_less);
            victims.pop_back();
            if (old >= 0) slot_map[old/n_expert][old % n_expert] = -1;
            slot_map[il][e] = s;
            key_of[s] = candidate;
            fills.push_back({e, s});
        }
        // Selection and placement are independent: retain the chosen victim set,
        // then place ascending logical experts in ascending freed slots. This
        // preserves coalesced uploads on cold banks despite demand-based ranking.
        std::sort(fills.begin(), fills.end(), [](const fill & a, const fill & b) { return a.expert < b.expert; });
        std::vector<int32_t> destinations;
        destinations.reserve(fills.size());
        for (const auto & f : fills) destinations.push_back(f.slot);
        std::sort(destinations.begin(), destinations.end());
        for (size_t i = 0; i < fills.size(); ++i) {
            fills[i].slot = destinations[i];
            slot_map[il][fills[i].expert] = destinations[i];
            key_of[destinations[i]] = il*n_expert + fills[i].expert;
        }
        std::sort(uniq.begin(), uniq.end());
        std::sort(temporary.begin(), temporary.end());
        return true;
    }
};
