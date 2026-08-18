from pathlib import Path

path = Path("tests/test-qwen38-codex-template.cpp")
text = path.read_text()
start_marker = "    const std::string generated_commentary =\n"
end_marker = "    const auto parsed_commentary = common_chat_parse(generated_commentary, false, phase_parser);\n"

start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0 or text.find(start_marker, start + 1) >= 0:
    raise RuntimeError("expected exactly one generated_commentary regression block")

replacement = r'''    const std::string generated_commentary =
        std::string("I should inspect the workspace.\n</think>\n") + commentary_marker + "\n"
        "I'll inspect the repository.\n\n"
        "<tool_call>\n"
        "<function=shell_command>\n"
        "<parameter=command>\n" +
        nlohmann::ordered_json("pwd").dump() +
        "\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
'''

text = text[:start] + replacement + text[end:]
path.write_text(text)
