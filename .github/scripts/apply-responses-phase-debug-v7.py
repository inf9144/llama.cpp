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
# Responses phases remain explicit model-side markers, but they are a separate
# protocol layer after Qwen's native </think> close.  Do not make the phase
# opener part of the reasoning delimiter and do not require a special newline
# shape between the two layers.
#
# Qwen's own <think>/<tool_call> framing is text-delimited and can collide with
# literal data.  Responses phase markers intentionally inherit that model
# protocol limitation rather than trying to "fix" it with a longer compound
# sentinel, which previously caused generation/runtime mismatches and runaway
# reasoning.
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
            // Keep Qwen's native reasoning parser for both runtime parsing and
            // generation. Responses phase metadata is parsed independently by
            // the following phase rule; it is not part of the </think> boundary.
            //
            // Like Qwen's native <think>/<tool_call> protocol, this is text
            // framing: a literal delimiter in generated data is inherently
            // ambiguous. Do not hide that model-level limitation behind a
            // longer compound sentinel.
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
        }
''',
    "restore native Qwen reasoning boundary",
)
chat = replace_in_qwen(
    chat,
'''    // Streamed output uses the collision-hardened tolerant parser. Fresh
    // Responses generation gets a separate strict grammar below.
''',
'''    // Runtime parsing keeps Qwen's native text framing. Fresh Responses
    // generation gets a separate strict grammar only to require phase metadata.
''',
    "native runtime parser comment",
)
chat_path.write_text(chat)


# ---------------------------------------------------------------------------
# Regression coverage now documents the actual contract:
# - fresh Responses generation must still use an eager grammar and emit phase;
# - runtime parsing keeps the legacy fallback for an unmarked item;
# - </think> and <response_phase> are independent, so legal whitespace between
#   them must not be baked into one compound delimiter.
# ---------------------------------------------------------------------------
test_path = Path("tests/test-qwen38-codex-template.cpp")
test = test_path.read_text()

test = replace_once(
    test,
'''    // Regression for a real runtime leak: the model may mention the literal
    // </think> delimiter while reasoning about its own protocol. Only the
    // close-tag immediately paired with the Responses phase opener may end
    // reasoning; the earlier literal must remain hidden reasoning text.
    const std::string generated_delimiter_collision =
        "The literal </think> token is being discussed, not emitted as a boundary.\\n"
        "Still reasoning after that literal token.\\n"
        "</think>\\n<response_phase>final_answer</response_phase>\\n"
        "Only this sentence is visible.";
    const auto parsed_delimiter_collision =
        common_chat_parse(generated_delimiter_collision, false, phase_parser);
    if (parsed_delimiter_collision.phase != "final_answer" ||
        parsed_delimiter_collision.content != "Only this sentence is visible." ||
        parsed_delimiter_collision.reasoning_content.find("literal </think> token") == std::string::npos ||
        parsed_delimiter_collision.reasoning_content.find("Still reasoning after") == std::string::npos ||
        parsed_delimiter_collision.content.find("</think>") != std::string::npos ||
        parsed_delimiter_collision.content.find("response_phase") != std::string::npos) {
        std::cerr << "Literal </think> inside reasoning leaked into visible Responses content\\n";
        return 1;
    }

''',
'''    // Qwen's native </think> delimiter is text framing, just like its
    // <tool_call> markers. A literal delimiter in generated reasoning is an
    // upstream model-protocol ambiguity and is intentionally not redefined by
    // the Responses phase extension.

''',
    "drop impossible reasoning collision guarantee",
)

test = replace_once(
    test,
'''    // Runtime parsing remains tolerant/collision-safe. Fresh generation is
    // strict through phase_params.grammar; malformed unmarked output must not
    // leak visible content from the runtime parser.
    bool unmarked_content_hidden = false;
    try {
        const auto parsed_unmarked_generation = common_chat_parse(
            "Legacy reasoning.\\n</think>\\n\\nLegacy final text.", false, phase_parser);
        unmarked_content_hidden =
            parsed_unmarked_generation.content.empty() &&
            parsed_unmarked_generation.tool_calls.empty();
    } catch (const std::exception &) {
        unmarked_content_hidden = true;
    }
    if (!unmarked_content_hidden) {
        std::cerr << "Responses runtime parser exposed unmarked generated content\\n";
        return 1;
    }
''',
'''    // Fresh generation is strict through phase_params.grammar, but runtime
    // parsing keeps the legacy fallback contract for already-produced/unmarked
    // material. The server resolves an empty phase structurally at egress.
    const auto parsed_unmarked_generation = common_chat_parse(
        "Legacy reasoning.\\n</think>\\n\\nLegacy final text.", false, phase_parser);
    if (!parsed_unmarked_generation.phase.empty() ||
        parsed_unmarked_generation.content != "Legacy final text.") {
        std::cerr << "Responses runtime parser lost the legacy phase fallback contract\\n";
        return 1;
    }
''',
    "restore runtime phase fallback regression",
)

test = replace_once(
    test,
'''    const std::string generated_final =
        "The task is complete.\\n</think>\\n<response_phase>final_answer</response_phase>\\nDone.";
    const auto parsed_final = common_chat_parse(generated_final, false, phase_parser);
    if (parsed_final.phase != "final_answer" || parsed_final.content != "Done." ||
        parsed_final.reasoning_content.find("response_phase") != std::string::npos) {
        std::cerr << "Qwen final_answer phase marker was not consumed as metadata\\n";
        return 1;
    }

''',
'''    const std::string generated_final =
        "The task is complete.\\n</think>\\n<response_phase>final_answer</response_phase>\\nDone.";
    const auto parsed_final = common_chat_parse(generated_final, false, phase_parser);
    if (parsed_final.phase != "final_answer" || parsed_final.content != "Done." ||
        parsed_final.reasoning_content.find("response_phase") != std::string::npos) {
        std::cerr << "Qwen final_answer phase marker was not consumed as metadata\\n";
        return 1;
    }

    // Phase is a separate layer after the native reasoning close. Do not bake
    // one particular whitespace spelling into the reasoning delimiter.
    const auto parsed_adjacent_phase = common_chat_parse(
        "Adjacent boundary.\\n</think><response_phase>final_answer</response_phase>\\nDone adjacent.",
        false,
        phase_parser);
    if (parsed_adjacent_phase.phase != "final_answer" ||
        parsed_adjacent_phase.content != "Done adjacent.") {
        std::cerr << "Qwen Responses parser coupled phase to a newline after </think>\\n";
        return 1;
    }

    const auto parsed_spaced_phase = common_chat_parse(
        "Spaced boundary.\\n</think>\\n\\n<response_phase>commentary</response_phase>\\nWorking.",
        false,
        phase_parser);
    if (parsed_spaced_phase.phase != "commentary" ||
        parsed_spaced_phase.content != "Working.") {
        std::cerr << "Qwen Responses parser rejected independent phase whitespace\\n";
        return 1;
    }

''',
    "independent phase boundary regressions",
)

test_path.write_text(test)
