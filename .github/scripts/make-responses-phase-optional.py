from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one marker in {path}, found {count}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "common/chat.cpp",
    '''        auto phase = p.eps();
        if (responses_phase_protocol) {
            phase = p.literal("<response_phase>") +
                    p.phase(p.literal("commentary") | p.literal("final_answer")) +
                    p.literal("</response_phase>") + p.space();
        }
''',
    '''        auto phase = p.eps();
        if (responses_phase_protocol) {
            auto phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("commentary") | p.literal("final_answer")) +
                p.literal("</response_phase>"));
            phase = p.optional(phase_block + p.space());
        }
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    auto no_think_inputs = phase_inputs;
''',
    '''    // The model signal is preferred, but omission must remain backward
    // compatible: the old Qwen parse still succeeds and leaves phase empty so
    // Responses egress can apply its structural fallback.
    const auto parsed_unmarked = common_chat_parse(
        "Legacy reasoning.\\n</think>\\n\\nLegacy final text.", false, phase_parser);
    if (!parsed_unmarked.phase.empty() || parsed_unmarked.content != "Legacy final text.") {
        std::cerr << "Unmarked Qwen output did not preserve Responses phase fallback\\n";
        return 1;
    }

    auto no_think_inputs = phase_inputs;
''',
)

# A final answer can legitimately produce both a reasoning item and a message
# item in Responses output. Assert the phase on the message item instead of
# assuming it is the sole (or first) output item.
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    const auto final_phase_response = final_phase_result.to_json_oaicompat_resp();
    if (!final_phase_response.contains("output") || final_phase_response.at("output").size() != 1 ||
        final_phase_response.at("output")[0].value("type", std::string()) != "message" ||
        final_phase_response.at("output")[0].value("phase", std::string()) != "final_answer") {
        std::cerr << "Responses egress lost the model-selected final_answer phase\\n";
        return 1;
    }
''',
    '''    const auto final_phase_response = final_phase_result.to_json_oaicompat_resp();
    bool saw_final_phase_nonstream = false;
    if (final_phase_response.contains("output") && final_phase_response.at("output").is_array()) {
        for (const auto & item : final_phase_response.at("output")) {
            if (item.value("type", std::string()) == "message") {
                saw_final_phase_nonstream = item.value("phase", std::string()) == "final_answer";
                break;
            }
        }
    }
    if (!saw_final_phase_nonstream) {
        std::cerr << "Responses egress lost the model-selected final_answer phase\\n";
        return 1;
    }
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''        if (event.value("event", std::string()) == "response.completed" &&
            event.contains("data") && event.at("data").contains("response") &&
            event.at("data").at("response").contains("output") &&
            !event.at("data").at("response").at("output").empty()) {
            saw_final_phase_completed =
                event.at("data").at("response").at("output")[0].value("phase", std::string()) == "final_answer";
        }
''',
    '''        if (event.value("event", std::string()) == "response.completed" &&
            event.contains("data") && event.at("data").contains("response") &&
            event.at("data").at("response").contains("output") &&
            event.at("data").at("response").at("output").is_array()) {
            for (const auto & item : event.at("data").at("response").at("output")) {
                if (item.value("type", std::string()) == "message") {
                    saw_final_phase_completed = item.value("phase", std::string()) == "final_answer";
                    break;
                }
            }
        }
''',
)

# Commentary with visible text plus a tool call also carries a reasoning item.
# Find the semantic output items by type instead of depending on array indices.
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    const auto commentary_phase_response = commentary_phase_result.to_json_oaicompat_resp();
    if (!commentary_phase_response.contains("output") || commentary_phase_response.at("output").size() != 2 ||
        commentary_phase_response.at("output")[0].value("type", std::string()) != "message" ||
        commentary_phase_response.at("output")[0].value("phase", std::string()) != "commentary" ||
        commentary_phase_response.at("output")[1].value("type", std::string()) != "function_call") {
        std::cerr << "Responses egress lost the model-selected commentary phase\\n";
        return 1;
    }
''',
    '''    const auto commentary_phase_response = commentary_phase_result.to_json_oaicompat_resp();
    bool saw_commentary_message = false;
    bool saw_commentary_tool_call = false;
    if (commentary_phase_response.contains("output") && commentary_phase_response.at("output").is_array()) {
        for (const auto & item : commentary_phase_response.at("output")) {
            const auto type = item.value("type", std::string());
            if (type == "message") {
                saw_commentary_message = item.value("phase", std::string()) == "commentary";
            } else if (type == "function_call") {
                saw_commentary_tool_call = true;
            }
        }
    }
    if (!saw_commentary_message || !saw_commentary_tool_call) {
        std::cerr << "Responses egress lost the model-selected commentary phase\\n";
        return 1;
    }
''',
)

# Lock down the early streaming contract too. Codex can act on the message phase
# before response.output_item.done, so the first message output_item.added event
# must already carry the parser-selected phase.
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    server_task_result_cmpl_final final_phase_result;
''',
    '''    task_result_state final_phase_stream_state(phase_parser, {}, false);
    server_task_result_cmpl_partial final_phase_partial;
    final_phase_partial.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
    final_phase_partial.content = generated_final;
    final_phase_partial.n_decoded = 1;
    final_phase_partial.update(final_phase_stream_state);
    bool saw_final_phase_added = false;
    for (const auto & event : final_phase_partial.to_json_oaicompat_resp()) {
        if (event.value("event", std::string()) == "response.output_item.added" &&
            event.contains("data") && event.at("data").contains("item") &&
            event.at("data").at("item").value("type", std::string()) == "message") {
            saw_final_phase_added =
                event.at("data").at("item").value("phase", std::string()) == "final_answer";
        }
    }
    if (!saw_final_phase_added) {
        std::cerr << "Streaming Responses output_item.added lost final_answer phase\\n";
        return 1;
    }

    server_task_result_cmpl_final final_phase_result;
''',
)
