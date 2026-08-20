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
# Codex semantics: commentary is a mid-turn assistant message and may exist
# with or without a tool call. A final_answer is terminal and therefore must
# never be followed by a tool call in the same model response.
#
# Keep the known-good v4 reasoning/runtime framing unchanged. Only strengthen
# the strict generation grammar so phase selection creates a small type-state:
#   commentary -> visible content, optionally followed by tool calls
#   final_answer -> visible content only; tool-call openers are forbidden
# For tool_choice=required, only the commentary branch exists.
# ---------------------------------------------------------------------------
chat_path = Path("common/chat.cpp")
chat = chat_path.read_text()

chat = replace_in_qwen(
    chat,
'''        auto phase = p.eps();
        if (responses_phase_protocol) {
            auto phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("commentary") | p.literal("final_answer")) +
                p.literal("</response_phase>"));
            // Runtime parsing stays tolerant for partial/legacy material, while
            // the separate generation grammar requires the phase structurally.
            phase = strict_phase_generation
                ? phase_block + p.space()
                : p.optional(phase_block + p.space());
        }
''',
'''        auto phase = p.eps();
        auto commentary_phase = p.eps();
        auto final_answer_phase = p.eps();
        if (responses_phase_protocol) {
            auto phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("commentary") | p.literal("final_answer")) +
                p.literal("</response_phase>"));
            auto commentary_phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("commentary")) +
                p.literal("</response_phase>"));
            auto final_answer_phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("final_answer")) +
                p.literal("</response_phase>"));
            // Runtime parsing stays tolerant for partial/legacy material, while
            // the separate generation grammar requires the phase structurally.
            phase = strict_phase_generation
                ? phase_block + p.space()
                : p.optional(phase_block + p.space());
            commentary_phase = commentary_phase_block + p.space();
            final_answer_phase = final_answer_phase_block + p.space();
        }
''',
    "add phase-specific strict parsers",
)

chat = replace_in_qwen(
    chat,
'''            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;

            // Qwen3-Coder models may occasionally omit the <tool_call> token.
            auto tool_call_body  = tool_choice + "</tool_call>" + p.space();
            auto tool_call_first = p.rule("tool-call-first", p.optional(p.literal("<tool_call>\\n")) + tool_call_body);
            auto tool_call       = p.rule("tool-call", "<tool_call>\\n" + tool_call_body);

            auto calls      = inputs.parallel_tool_calls ? tool_call_first + p.zero_or_more(tool_call) : tool_call_first;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(calls, min_calls, 1));

            return generation_prompt +
                   (reasoning << phase << p.content(p.until_one_of(tool_call_starts)) << tool_calls);
''',
'''            // Qwen3-Coder models may occasionally omit the <tool_call> token.
            auto tool_call_body  = tool_choice + "</tool_call>" + p.space();
            auto tool_call_first = p.rule("tool-call-first", p.optional(p.literal("<tool_call>\\n")) + tool_call_body);
            auto tool_call       = p.rule("tool-call", "<tool_call>\\n" + tool_call_body);

            auto calls = inputs.parallel_tool_calls ? tool_call_first + p.zero_or_more(tool_call) : tool_call_first;

            if (responses_phase_protocol && strict_phase_generation) {
                auto required_tool_calls = p.trigger_rule("tool-call-root", p.repeat(calls, 1, 1));
                auto content_before_tool = p.content(p.until_one_of(tool_call_starts));

                if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                    // A required tool call is always a mid-turn commentary item.
                    return generation_prompt +
                           (reasoning << commentary_phase << content_before_tool << required_tool_calls);
                }

                // With tool_choice=auto, commentary is valid both as a pure
                // progress/preamble message and as a message followed by tools.
                // final_answer is also valid, but only if no recognized tool
                // opener appears later in the same generated item.
                auto commentary_item = commentary_phase << content_before_tool << p.optional(required_tool_calls);
                auto final_answer_item = final_answer_phase << p.content(p.until_one_of(tool_call_starts)) << p.end();
                return generation_prompt + (reasoning << (commentary_item | final_answer_item));
            }

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(calls, min_calls, 1));

            return generation_prompt +
                   (reasoning << phase << p.content(p.until_one_of(tool_call_starts)) << tool_calls);
''',
    "constrain strict Responses tool-call phase",
)

chat_path.write_text(chat)


# ---------------------------------------------------------------------------
# Defensive egress invariant. The strict grammar should prevent this state for
# fresh Qwen generation, but historical/non-participating producers may still
# supply final_answer + tool_calls. Codex requires such a message to behave as
# commentary so its Working indicator and mid-turn lifecycle are preserved.
# ---------------------------------------------------------------------------
server_task_path = Path("tools/server/server-task.cpp")
server_task = server_task_path.read_text()
server_task = replace_once(
    server_task,
'''static std::string server_task_response_message_phase(const common_chat_msg & msg) {
    if (msg.phase == "commentary" || msg.phase == "final_answer") {
        return msg.phase;
    }
    return msg.tool_calls.empty() ? "final_answer" : "commentary";
}
''',
'''static std::string server_task_response_message_phase(const common_chat_msg & msg) {
    // A Responses message that accompanies tool calls is necessarily mid-turn.
    // Prefer this structural invariant over a contradictory model marker.
    if (!msg.tool_calls.empty()) {
        return "commentary";
    }
    if (msg.phase == "commentary" || msg.phase == "final_answer") {
        return msg.phase;
    }
    return "final_answer";
}
''',
    "normalize tool-call phase at Responses egress",
)
server_task_path.write_text(server_task)


# ---------------------------------------------------------------------------
# Regression coverage for the defensive invariant and for commentary without
# tools. The live generation grammar is additionally exercised by the focused
# Qwen build/runtime test and the subsequent Codex live test.
# ---------------------------------------------------------------------------
test_path = Path("tests/test-qwen38-codex-template.cpp")
test = test_path.read_text()

test = replace_once(
    test,
'''    const auto parsed_final = common_chat_parse(generated_final, false, phase_parser);
    if (parsed_final.phase != "final_answer" || parsed_final.content != "Done." ||
        parsed_final.reasoning_content.find("response_phase") != std::string::npos) {
        std::cerr << "Qwen final_answer phase marker was not consumed as metadata\\n";
        return 1;
    }

''',
'''    const auto parsed_final = common_chat_parse(generated_final, false, phase_parser);
    if (parsed_final.phase != "final_answer" || parsed_final.content != "Done." ||
        parsed_final.reasoning_content.find("response_phase") != std::string::npos) {
        std::cerr << "Qwen final_answer phase marker was not consumed as metadata\\n";
        return 1;
    }

    const auto parsed_commentary_without_tool = common_chat_parse(
        "Still working.\\n</think>\\n<response_phase>commentary</response_phase>\\nProgress update.",
        false,
        phase_parser);
    if (parsed_commentary_without_tool.phase != "commentary" ||
        parsed_commentary_without_tool.content != "Progress update." ||
        !parsed_commentary_without_tool.tool_calls.empty()) {
        std::cerr << "Qwen Responses parser lost commentary-without-tool semantics\\n";
        return 1;
    }

''',
    "commentary without tool regression",
)

test = replace_once(
    test,
'''    // Backward compatibility: an unmarked parser result still gets the old
    // structural fallback, so non-participating templates/models keep working.
    common_chat_msg fallback_msg = message("assistant", "Legacy preamble.");
''',
'''    // Defensive semantic normalization: even if a producer contradicts the
    // protocol and marks a message with tools as final_answer, Responses egress
    // must expose it as commentary so Codex keeps the turn in progress.
    common_chat_msg contradictory_phase_msg = parsed_commentary;
    contradictory_phase_msg.phase = "final_answer";
    server_task_result_cmpl_final contradictory_phase_result;
    contradictory_phase_result.oaicompat_model = "test-model";
    contradictory_phase_result.oaicompat_msg = contradictory_phase_msg;
    const auto contradictory_phase_response = contradictory_phase_result.to_json_oaicompat_resp();
    bool normalized_contradictory_phase = false;
    if (contradictory_phase_response.contains("output") &&
        contradictory_phase_response.at("output").is_array()) {
        for (const auto & item : contradictory_phase_response.at("output")) {
            if (item.value("type", std::string()) == "message") {
                normalized_contradictory_phase =
                    item.value("phase", std::string()) == "commentary";
                break;
            }
        }
    }
    if (!normalized_contradictory_phase) {
        std::cerr << "Responses egress preserved invalid final_answer + tool_calls state\\n";
        return 1;
    }

    // Backward compatibility: an unmarked parser result still gets the old
    // structural fallback, so non-participating templates/models keep working.
    common_chat_msg fallback_msg = message("assistant", "Legacy preamble.");
''',
    "contradictory phase normalization regression",
)

test_path.write_text(test)
