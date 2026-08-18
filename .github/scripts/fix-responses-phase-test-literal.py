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

old_history = r'''        {"input", nlohmann::ordered_json::array({
            {
                {"type", "message"},
                {"role", "assistant"},
                {"status", "completed"},
                {"phase", "commentary"},
                {"content", nlohmann::ordered_json::array({
                    {{"type", "output_text"}, {"text", "Checking the repository."}},
                })},
            },
        })},
'''
new_history = r'''        {"input", nlohmann::ordered_json::array({
            {
                {"type", "message"},
                {"role", "user"},
                {"content", nlohmann::ordered_json::array({
                    {{"type", "input_text"}, {"text", "Inspect the repository."}},
                })},
            },
            {
                {"type", "message"},
                {"role", "assistant"},
                {"status", "completed"},
                {"phase", "commentary"},
                {"content", nlohmann::ordered_json::array({
                    {{"type", "output_text"}, {"text", "Checking the repository."}},
                })},
            },
        })},
'''
if text.count(old_history) != 1:
    raise RuntimeError("expected exactly one assistant-only Responses phase history fixture")
text = text.replace(old_history, new_history, 1)

old_assert = r'''        converted_phase_history.at("messages").size() != 1 ||
        converted_phase_history.at("messages")[0].value("phase", std::string()) != "commentary") {
'''
new_assert = r'''        converted_phase_history.at("messages").size() != 2 ||
        converted_phase_history.at("messages")[1].value("phase", std::string()) != "commentary") {
'''
if text.count(old_assert) != 1:
    raise RuntimeError("expected exactly one Responses phase history assertion")
text = text.replace(old_assert, new_assert, 1)

path.write_text(text)
