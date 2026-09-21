import pytest
from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.cache_ssd_dir = str(tmp_path / "ssd-cache")
    # small RAM cache to force eviction to SSD
    server.cache_ram = 4
    server.n_ctx = 512


def test_ssd_hit_after_ram_eviction():
    global server
    server.start()

    # first prompt, fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 0

    # a long filler prompt to evict the first one from the small RAM cache
    filler = " filler " * 40
    res = server.make_request("POST", "/completion", data={
        "prompt": filler,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # re-send the first prompt: it should be restored from the SSD cache
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France? ",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    # most of the prompt is reused from the SSD cache
    assert res.body["timings"]["prompt_n"] < prompt_n_full


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
    server.cache_ram = 4
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
