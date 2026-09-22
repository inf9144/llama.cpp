import pytest
from utils import *

import os
import struct

server = ServerPreset.tinyllama2()


def get_metric(name):
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith("llamacpp:" + name + " "):
            return float(line.split()[-1])
    return 0.0


FP_OFFSET = 16   # after magic(4) + version(4) + last_access_ms(8)
FP_SIZE = 168    # sizeof(server_prompt_cache_ssd_fingerprint)


def read_fingerprints():
    # read the full fingerprint from every .lsc file. all files share the same
    # fingerprint (same model + context params), so any file is a valid sample
    fps = []
    for root, dirs, files in os.walk(server.cache_ssd_dir):
        for f in files:
            if not f.endswith(".lsc"):
                continue
            with open(os.path.join(root, f), "rb") as fh:
                data = fh.read()
            magic = struct.unpack("<I", data[0:4])[0]
            assert magic == 0x5343504c, f"bad magic {magic:#x}"
            fps.append(data[FP_OFFSET:FP_OFFSET + FP_SIZE])
    return fps


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.cache_ssd_dir = str(tmp_path / "ssd-cache")
    # small RAM cache to force eviction to SSD
    server.cache_ram = 1
    server.n_ctx = 512
    server.server_metrics = True


def test_ssd_hit_after_ram_eviction():
    global server
    server.start()

    # prompt A, fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # fill the RAM cache with distinct prompts, each overwriting slot 0,
    # so prompt A is evicted from RAM to SSD
    for i in range(20):
        res = server.make_request("POST", "/completion", data={
            "prompt": f"Filler prompt number {i} with some extra text to take space. ",
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200

    # re-send prompt A: it must be restored from the SSD cache
    hits_before = get_metric("prompt_cache_ssd_hits_total")
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    hits_after = get_metric("prompt_cache_ssd_hits_total")
    assert hits_after > hits_before, f"expected an SSD hit, before={hits_before} after={hits_after}"


def test_ssd_hit_survives_touch_and_restart():
    # a successful SSD hit calls touch_file(); the file must stay valid so the
    # same entry can be restored again. verify the fingerprint is not corrupted
    # and the entry survives a restart.
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # evict prompt A to SSD
    for i in range(20):
        res = server.make_request("POST", "/completion", data={
            "prompt": f"Filler prompt number {i} with some extra text to take space. ",
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200

    # restore A from SSD (this calls touch_file)
    hits_before = get_metric("prompt_cache_ssd_hits_total")
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    hits_after = get_metric("prompt_cache_ssd_hits_total")
    assert hits_after > hits_before, "expected an SSD hit after eviction"

    # the full fingerprint must be intact in every file after touch_file
    fps_before = read_fingerprints()
    assert fps_before, "no .lsc file found"
    for fp in fps_before:
        n_layer = struct.unpack("<i", fp[0:4])[0]
        assert n_layer > 0, f"corrupt n_layer {n_layer} (touch_file bug)"

    # restart the server with the same SSD directory
    server.stop()
    server.start()

    # the fingerprint must be valid and unchanged after the restart (the
    # shutdown flush re-saves the state with a fresh fingerprint)
    fps_after = read_fingerprints()
    assert fps_after, "no .lsc file found after restart"
    for fp in fps_after:
        n_layer = struct.unpack("<i", fp[0:4])[0]
        assert n_layer > 0, f"corrupt n_layer {n_layer} after restart"
    assert set(fps_after) == set(fps_before), "fingerprint changed after restart"

    # the entry must be restorable again after the restart
    hits_before2 = get_metric("prompt_cache_ssd_hits_total")
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    hits_after2 = get_metric("prompt_cache_ssd_hits_total")
    assert hits_after2 > hits_before2, \
        f"expected an SSD hit after restart, before={hits_before2} after={hits_after2}"


def test_ssd_restore_after_restart(tmp_path):
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 0

    # restart the server with the same SSD directory
    server.stop()
    server.start()

    # the state should be restored from the SSD cache after the restart
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] < prompt_n_full


def test_large_state_exceeding_ram_limit_reaches_ssd(tmp_path):
    # a single state larger than the RAM limit must still be saved to SSD, even
    # though it can never fit in the RAM tier. tinygemma3 has a large enough KV
    # cache that a short prompt already exceeds the 1 MiB RAM limit.
    global server
    server = ServerPreset.tinygemma3()
    server.temperature = 0.0
    server.cache_ssd_dir = str(tmp_path / "ssd-cache")
    server.cache_ram = 1
    server.n_ctx = 512
    server.server_metrics = True
    server.start()

    # a prompt whose KV state exceeds the 1 MiB RAM limit
    long_prompt = "This prompt has enough text to make the KV state exceed the one MiB RAM cache limit. " * 4

    res = server.make_request("POST", "/completion", data={
        "prompt": long_prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # replace the slot: the current state is too large for RAM, so it must be
    # saved directly to SSD
    res = server.make_request("POST", "/completion", data={
        "prompt": "A completely different prompt to replace the slot. ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    ssd_size = get_metric("prompt_cache_ssd_size_bytes")
    assert ssd_size > 0, f"expected the large state to be saved to SSD, size={ssd_size}"

    # re-send the long prompt: it must be restored from SSD
    hits_before = get_metric("prompt_cache_ssd_hits_total")
    res = server.make_request("POST", "/completion", data={
        "prompt": long_prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    hits_after = get_metric("prompt_cache_ssd_hits_total")
    assert hits_after > hits_before, \
        f"expected an SSD hit for the large state, before={hits_before} after={hits_after}"


def test_no_ssd_config_unchanged(tmp_path):
    # without the SSD config, the behavior is unchanged (no SSD cache)
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.cache_ram = 1
    server.n_ctx = 512
    # no cache_ssd_dir
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 0

    # the SSD directory should not have been created
    assert not (tmp_path / "ssd-cache").exists()


def test_growing_prompt_same_slot_no_full_refill():
    # a prompt that grows on the same slot is a continuation: the slot is
    # reused as-is and only the delta is prefilled, no full re-prefill
    global server
    server.start()

    base = "What is the capital of France? "
    res = server.make_request("POST", "/completion", data={
        "prompt": base,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_base = res.body["timings"]["prompt_n"]
    assert prompt_n_base > 0

    # extend the prompt: only the new tokens are prefilled
    res = server.make_request("POST", "/completion", data={
        "prompt": base + " And of Germany? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] < prompt_n_base

    # extend again: still only the delta
    res = server.make_request("POST", "/completion", data={
        "prompt": base + " And of Germany? And of Italy? ",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] < prompt_n_base
