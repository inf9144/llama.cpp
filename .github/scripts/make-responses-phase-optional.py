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
