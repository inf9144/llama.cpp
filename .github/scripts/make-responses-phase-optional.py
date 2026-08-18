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
