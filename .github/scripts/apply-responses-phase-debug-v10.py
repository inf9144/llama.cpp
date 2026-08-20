from pathlib import Path


def replace_in_function(text: str, start_marker: str, end_marker: str, old: str, new: str, label: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f"{label}: function start not found")
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise SystemExit(f"{label}: function end marker not found")
    section = text[start:end]
    count = section.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    section = section.replace(old, new, 1)
    return text[:start] + section + text[end:]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Codex 0.148 consumes response.completed.response.end_turn as an optional
# provider-directed continuation signal. Some(false) forces another sampling
# step even when there is no tool call. A commentary message is explicitly
# mid-turn, while final_answer is terminal, so the Responses phase is the
# authoritative source for this signal.
#
# Emit only end_turn=false for commentary. Omit the field for terminal/fallback
# responses so the normal Responses wire shape stays unchanged; Codex treats
# omitted and true equivalently for turn completion.
# ---------------------------------------------------------------------------
server_task_path = Path("tools/server/server-task.cpp")
server_task = server_task_path.read_text()

server_task = replace_in_function(
    server_task,
    "json server_task_result_cmpl_final::to_json_oaicompat_resp() {",
    "json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {",
'''    };

    return res;
''',
'''    };

    if (server_task_response_message_phase(msg) == "commentary") {
        res["end_turn"] = false;
    }

    return res;
''',
    "non-stream Responses commentary end_turn",
)

server_task = replace_in_function(
    server_task,
    "json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {",
    "json server_task_result_cmpl_final::to_json_oaicompat_asr() {",
'''    if (stats.is_set()) {
        server_sent_events.back().at("data").push_back({"timings", stats.to_json()});
    }
''',
'''    const std::string resolved_phase = server_task_response_message_phase(oaicompat_msg);
    if (resolved_phase == "commentary") {
        server_sent_events.back().at("data").at("response")["end_turn"] = false;
    }
    SRV_INF(
        "[responses-phase] event=response.completed model_phase=%s resolved_phase=%s end_turn=%s tool_calls=%zu\\n",
        oaicompat_msg.phase.empty() ? "<none>" : oaicompat_msg.phase.c_str(),
        resolved_phase.c_str(),
        resolved_phase == "commentary" ? "false" : "<omitted>",
        oaicompat_msg.tool_calls.size());

    if (stats.is_set()) {
        server_sent_events.back().at("data").push_back({"timings", stats.to_json()});
    }
''',
    "stream Responses commentary end_turn",
)

server_task_path.write_text(server_task)


# ---------------------------------------------------------------------------
# Deterministic regression coverage. This proves the important case that could
# not be forced reliably in a live model run: commentary with no tool call must
# complete the current HTTP response with end_turn=false, causing Codex to make
# another sampling request. final_answer deliberately leaves end_turn omitted.
# ---------------------------------------------------------------------------
test_path = Path("tests/test-qwen38-codex-template.cpp")
test = test_path.read_text()

test = replace_once(
    test,
'''    if (parsed_commentary_without_tool.phase != "commentary" ||
        parsed_commentary_without_tool.content != "Progress update." ||
        !parsed_commentary_without_tool.tool_calls.empty()) {
        std::cerr << "Qwen Responses parser lost commentary-without-tool semantics\\n";
        return 1;
    }

''',
'''    if (parsed_commentary_without_tool.phase != "commentary" ||
        parsed_commentary_without_tool.content != "Progress update." ||
        !parsed_commentary_without_tool.tool_calls.empty()) {
        std::cerr << "Qwen Responses parser lost commentary-without-tool semantics\\n";
        return 1;
    }

    server_task_result_cmpl_final commentary_without_tool_result;
    commentary_without_tool_result.oaicompat_model = "test-model";
    commentary_without_tool_result.oai_resp_id = "resp_commentary_without_tool";
    commentary_without_tool_result.oai_resp_message_id = "msg_commentary_without_tool";
    commentary_without_tool_result.n_prompt_tokens = 0;
    commentary_without_tool_result.n_prompt_tokens_cache = 0;
    commentary_without_tool_result.n_decoded = 0;
    commentary_without_tool_result.oaicompat_msg = parsed_commentary_without_tool;
    commentary_without_tool_result.oaicompat_msg.role = "assistant";

    const auto commentary_without_tool_response =
        commentary_without_tool_result.to_json_oaicompat_resp();
    if (!commentary_without_tool_response.contains("end_turn") ||
        commentary_without_tool_response.at("end_turn") != false) {
        std::cerr << "Responses non-stream commentary without tool did not request follow-up\\n";
        return 1;
    }

    bool saw_commentary_without_tool_end_turn_false = false;
    for (const auto & event : commentary_without_tool_result.to_json_oaicompat_resp_stream()) {
        if (event.value("event", std::string()) == "response.completed" &&
            event.contains("data") && event.at("data").contains("response")) {
            const auto & response = event.at("data").at("response");
            saw_commentary_without_tool_end_turn_false =
                response.contains("end_turn") && response.at("end_turn") == false;
        }
    }
    if (!saw_commentary_without_tool_end_turn_false) {
        std::cerr << "Responses streaming commentary without tool did not emit end_turn=false\\n";
        return 1;
    }

''',
    "commentary without tool end_turn regression",
)

test = replace_once(
    test,
'''    if (!saw_final_phase_nonstream) {
        std::cerr << "Responses egress lost the model-selected final_answer phase\\n";
        return 1;
    }

''',
'''    if (!saw_final_phase_nonstream) {
        std::cerr << "Responses egress lost the model-selected final_answer phase\\n";
        return 1;
    }
    if (final_phase_response.contains("end_turn")) {
        std::cerr << "Responses final_answer unexpectedly emitted end_turn override\\n";
        return 1;
    }

''',
    "final answer omits end_turn override",
)

test = replace_once(
    test,
'''    if (!normalized_contradictory_phase) {
        std::cerr << "Responses egress preserved invalid final_answer + tool_calls state\\n";
        return 1;
    }

''',
'''    if (!normalized_contradictory_phase) {
        std::cerr << "Responses egress preserved invalid final_answer + tool_calls state\\n";
        return 1;
    }
    if (!contradictory_phase_response.contains("end_turn") ||
        contradictory_phase_response.at("end_turn") != false) {
        std::cerr << "Responses contradictory tool-call phase did not request follow-up\\n";
        return 1;
    }

''',
    "contradictory tool phase end_turn regression",
)

test_path.write_text(test)
