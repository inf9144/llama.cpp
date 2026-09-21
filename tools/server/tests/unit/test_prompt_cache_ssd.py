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
    # same entry can be restored again. verify the fingerprint is not corrupted.
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

    # the file must still be valid after touch_file: the fingerprint is intact
    found = False
    for root, dirs, files in os.walk(server.cache_ssd_dir):
        for f in files:
            if not f.endswith(".lsc"):
                continue
            with open(os.path.join(root, f), "rb") as fh:
                data = fh.read()
            magic = struct.unpack("<I", data[0:4])[0]
            assert magic == 0x5343504c, f"bad magic {magic:#x}"
            # n_layer sits at offset 16 (after magic, version, last_access_ms)
            n_layer = struct.unpack("<i", data[16:20])[0]
            assert 0 < n_layer < 1000, f"corrupt n_layer {n_layer} (touch_file bug)"
            found = True
    assert found, "no .lsc file found"


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
