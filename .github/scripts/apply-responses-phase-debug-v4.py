from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_in_qwen(text: str, old: str, new: str, label: str) -> str:
    marker = "static common_chat_params common_chat_params_init_qwen3_coder("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"{label}: qwen3 coder function not found")
    end = text.find("\nstatic common_chat_params ", start + len(marker))
    if end < 0:
        raise SystemExit(f"{label}: next chat parser function not found")
    section = text[start:end]
    count = section.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one qwen match, found {count}")
    section = section.replace(old, new, 1)
    return text[:start] + section + text[end:]


# ---------------------------------------------------------------------------
# Canonicalize fresh generation at </think> while keeping runtime parsing
# tolerant to already-legal whitespace variants between </think> and phase.
# ---------------------------------------------------------------------------
chat_path = Path("common/chat.cpp")
chat = chat_path.read_text()
chat = replace_in_qwen(
    chat,
'''        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            if (responses_phase_protocol && !strict_phase_generation) {
                // Runtime parser hardening only: a literal </think> mentioned in
                // reasoning is not a boundary unless a phase opener follows it.
                // This protects streaming from leaking the remaining reasoning.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until("</think>\\n<response_phase>")) +
                                       p.literal("</think>") + p.space());
            } else {
                // Generation grammar keeps Qwen's native reasoning boundary.
                // The required phase parser follows this close independently.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                       (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
            }
        }
''',
'''        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            if (responses_phase_protocol && !strict_phase_generation) {
                // Runtime parser hardening only: pair the real reasoning close
                // with the following phase opener, but accept whitespace forms
                // that earlier strict grammars could legally generate.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({
                                           "</think><response_phase>",
                                           "</think> <response_phase>",
                                           "</think>\\t<response_phase>",
                                           "</think>\\n<response_phase>",
                                           "</think>\\n\\n<response_phase>",
                                       })) +
                                       p.literal("</think>") + p.space());
            } else if (responses_phase_protocol && strict_phase_generation) {
                // Fresh Responses generation uses Qwen's native </think> close,
                // followed by one canonical newline before the required phase.
                // Matching </think>\\n also avoids treating an inline literal
                // </think> mention as the reasoning boundary.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until("</think>\\n")) +
                                       p.literal("</think>\\n"));
            } else {
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                       (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
            }
        }
''',
    "phase whitespace boundary",
)
chat_path.write_text(chat)


# ---------------------------------------------------------------------------
# If final parsing ever produces no semantic message again, make the failure
# visible in the server log instead of leaving only the slot-release line.
# ---------------------------------------------------------------------------
server_task_path = Path("tools/server/server-task.cpp")
server_task = server_task_path.read_text()
server_task = replace_once(
    server_task,
'''    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
''',
'''    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!is_partial && new_msg.empty() && !generated_text.empty()) {
        SRV_INF(
            "[responses-phase] parser final-empty generated_chars=%zu think_close=%s phase_open=%s tool_call=%s\\n",
            generated_text.size(),
            generated_text.find("</think>") != std::string::npos ? "yes" : "no",
            generated_text.find("<response_phase>") != std::string::npos ? "yes" : "no",
            generated_text.find("<tool_call>") != std::string::npos ? "yes" : "no");
    }
    if (!new_msg.empty()) {
''',
    "final empty parser diagnostic",
)
server_task_path.write_text(server_task)


# ---------------------------------------------------------------------------
# Regression: runtime parsing must accept the no-whitespace form that the
# previous eager grammar permitted, plus a double-newline variant.
# ---------------------------------------------------------------------------
test_path = Path("tests/test-qwen38-codex-template.cpp")
test = test_path.read_text()
test = replace_once(
    test,
'''    // Exercise partial parsing of the internal marker. No marker prefix may
    // escape into a streamed reasoning/content delta while it is incomplete.
''',
'''    const std::string generated_commentary_adjacent =
        std::string("Adjacent phase boundary.\\n</think>") + commentary_marker + "\\n"
        "<tool_call>\\n"
        "<function=shell_command>\\n"
        "<parameter=command>\\n" +
        nlohmann::ordered_json("pwd").dump() +
        "\\n</parameter>\\n"
        "</function>\\n"
        "</tool_call>";
    const auto parsed_commentary_adjacent =
        common_chat_parse(generated_commentary_adjacent, false, phase_parser);
    if (parsed_commentary_adjacent.phase != "commentary" ||
        parsed_commentary_adjacent.tool_calls.size() != 1) {
        std::cerr << "Qwen runtime parser rejected adjacent </think><response_phase> boundary\\n";
        return 1;
    }

    const std::string generated_final_double_newline =
        "Double newline boundary.\\n</think>\\n\\n<response_phase>final_answer</response_phase>\\nDone.";
    const auto parsed_final_double_newline =
        common_chat_parse(generated_final_double_newline, false, phase_parser);
    if (parsed_final_double_newline.phase != "final_answer" ||
        parsed_final_double_newline.content != "Done.") {
        std::cerr << "Qwen runtime parser rejected double-newline Responses phase boundary\\n";
        return 1;
    }

    // Exercise partial parsing of the internal marker. No marker prefix may
    // escape into a streamed reasoning/content delta while it is incomplete.
''',
    "phase whitespace regression tests",
)
test_path.write_text(test)
