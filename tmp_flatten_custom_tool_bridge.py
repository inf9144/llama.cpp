from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}\n--- old ---\n{old}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "tools/server/server-chat.cpp",
    '''        description += "This is a Responses custom/freeform tool. When invoking it through this model interface, place the raw freeform payload verbatim in the `data` string inside the single `input` object. The `input.data` wrapper is transport-only; the Responses API receives only the decoded raw string.";
''',
    '''        description += "This is a Responses custom/freeform tool. When invoking it through this model interface, place the complete raw freeform payload in the single `input` string argument. Do not wrap it in another object. String arguments are JSON-encoded by the model-facing XML transport so embedded newlines, quotes, backslashes, and XML-like delimiter text remain data. The Responses API receives the decoded raw string.";
''',
)

replace_once(
    "tools/server/server-chat.cpp",
    '''                        {"input", json {
                            {"type", "object"},
                            {"description", "Transport wrapper for a Responses custom/freeform payload."},
                            {"properties", json {
                                {"data", json {
                                    {"type", "string"},
                                    {"description", "Raw freeform input passed verbatim to the custom tool."},
                                }},
                            }},
                            {"required", json::array({"data"})},
                            {"additionalProperties", false},
                        }},
''',
    '''                        {"input", json {
                            {"type", "string"},
                            {"description", "Complete raw freeform input passed verbatim to the custom tool."},
                        }},
''',
)

replace_once(
    "tools/server/server-chat.cpp",
    '''                // Responses custom/freeform calls use a nested JSON transport so
                // arbitrary payload text cannot collide with XML parameter framing.
''',
    '''                // Responses custom/freeform calls use one JSON-string transport argument.
                // The chat template JSON-encodes string parameters so arbitrary payload text
                // cannot collide with XML parameter framing.
''',
)

replace_once(
    "tools/server/server-chat.cpp",
    '''                        {"arguments", json({{"input", json({{"data", item.at("input")}})}}).dump()},
''',
    '''                        {"arguments", json({{"input", item.at("input")}}).dump()},
''',
)

replace_once(
    "tools/server/server-task.cpp",
    '''    if (!arguments.is_object() || arguments.size() != 1 ||
        !arguments.contains("input") || !arguments.at("input").is_object()) {
        throw std::runtime_error(
            "Responses custom tool '" + tool_call.name + "' must produce exactly one 'input' transport object");
    }

    const json & input = arguments.at("input");
    if (input.size() != 1 || !input.contains("data") || !input.at("data").is_string()) {
        throw std::runtime_error(
            "Responses custom tool '" + tool_call.name + "' must produce exactly one string 'input.data' argument");
    }

    return input.at("data").get<std::string>();
''',
    '''    if (!arguments.is_object() || arguments.size() != 1 ||
        !arguments.contains("input") || !arguments.at("input").is_string()) {
        throw std::runtime_error(
            "Responses custom tool '" + tool_call.name + "' must produce exactly one string 'input' argument");
    }

    return arguments.at("input").get<std::string>();
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    // Responses custom/freeform tools wrap the raw payload in input.data. The
    // outer object forces Qwen3-Coder onto its JSON-value PEG path, while the
    // nested string schema still guarantees that the transported value is text.
    common_chat_tool custom_tool;
    custom_tool.name = "apply_patch";
    custom_tool.description = "Responses custom/freeform transport test";
    custom_tool.parameters = R"({"type":"object","properties":{"input":{"type":"object","properties":{"data":{"type":"string"}},"required":["data"],"additionalProperties":false}},"required":["input"],"additionalProperties":false})";
''',
    '''    // Responses custom/freeform tools use one direct string argument. The
    // llama.cpp:xml-string-args=json template marker forces that string through
    // the JSON-value PEG path, so embedded XML delimiter text remains data.
    common_chat_tool custom_tool;
    custom_tool.name = "apply_patch";
    custom_tool.description = "Responses custom/freeform transport test";
    custom_tool.parameters = R"({"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false})";
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    const std::string encoded_custom_input = json({{"data", custom_input}}).dump();
''',
    '''    const std::string encoded_custom_input = json(custom_input).dump();
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    if (!parsed_custom_arguments.contains("input") ||
        !parsed_custom_arguments.at("input").is_object() ||
        !parsed_custom_arguments.at("input").contains("data") ||
        !parsed_custom_arguments.at("input").at("data").is_string() ||
        parsed_custom_arguments.at("input").at("data").get<std::string>() != custom_input) {
        std::cerr << "Custom/freeform PEG transport corrupted literal XML delimiter content\\n";
        return 1;
    }
''',
    '''    if (!parsed_custom_arguments.contains("input") ||
        !parsed_custom_arguments.at("input").is_string() ||
        parsed_custom_arguments.at("input").get<std::string>() != custom_input) {
        std::cerr << "Custom/freeform PEG transport corrupted literal XML delimiter content\\n";
        return 1;
    }
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''        json({{"input", {{"data", custom_input}}}}).dump(),
''',
    '''        json({{"input", custom_input}}).dump(),
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    if (!historical_value.is_object() || historical_value.size() != 1 ||
        !historical_value.contains("data") || !historical_value.at("data").is_string() ||
        historical_value.at("data").get<std::string>() != custom_input) {
        std::cerr << "Historical custom/freeform replay corrupted the transported payload\\n";
        return 1;
    }
''',
    '''    if (!historical_value.is_string() || historical_value.get<std::string>() != custom_input) {
        std::cerr << "Historical custom/freeform replay corrupted the transported payload\\n";
        return 1;
    }
''',
)

# Verify the Responses ingress bridge itself now exposes a direct string input.
needle = '''    const json tool_search_request = {\n'''
insert = '''    const json custom_tool_request = {
        {"model", "test-model"},
        {"input", "Apply a patch"},
        {"tools", json::array({
            {
                {"type", "custom"},
                {"name", "apply_patch"},
                {"description", "Apply a patch"},
            },
        })},
    };
    const auto converted_custom_tool = server_chat_convert_responses_to_chatcmpl(custom_tool_request);
    if (!converted_custom_tool.contains("tools") || converted_custom_tool.at("tools").size() != 1 ||
        converted_custom_tool.at("tools")[0]["function"]["parameters"]["properties"]["input"].value("type", std::string()) != "string") {
        std::cerr << "Responses custom/freeform bridge did not expose direct string input\\n";
        return 1;
    }

'''
p = Path("tests/test-qwen38-codex-template.cpp")
text = p.read_text()
if text.count(needle) != 1:
    raise SystemExit("test insertion point not unique")
p.write_text(text.replace(needle, insert + needle, 1))
