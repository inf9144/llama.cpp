from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "tools/server/server-chat.cpp",
    '''        description += "This is a Responses custom/freeform tool. When invoking it through this model interface, place the complete raw freeform payload in the single `input` string argument. Do not wrap it in another object. String arguments are JSON-encoded by the model-facing XML transport so embedded newlines, quotes, backslashes, and XML-like delimiter text remain data. The Responses API receives the decoded raw string.";
''',
    '''        description += "This is a Responses custom/freeform tool. When invoking it through this model interface, place the complete raw freeform payload in the single `input` string argument. Do not wrap it in another object. String arguments are JSON-encoded by the model-facing XML transport so embedded newlines, quotes, backslashes, and XML-like delimiter text remain data. The Responses API receives the decoded raw string.";

        if (custom_tool.at("name").get<std::string>() == "apply_patch") {
            description +=
                "\\n\\nCodex apply_patch syntax: use `*** Begin Patch`, `*** Update File: <path>`, and `*** End Patch`; "
                "do not emit standard unified-diff range headers such as `@@ -1,3 +1,4 @@`. "
                "For update hunks, the first hunk may begin directly with diff lines, and `@@` or `@@ <context>` may be used as Codex context markers when needed. "
                "Every file-content line in an update hunk must start with a space for context, `+` for an added line, or `-` for a removed line. "
                "A literal file-content line beginning with `***` must still carry its diff prefix, for example `+*** literal content` when inserting it.";
        }
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    if (!converted_custom_tool.contains("tools") || converted_custom_tool.at("tools").size() != 1 ||
        converted_custom_tool.at("tools")[0]["function"]["parameters"]["properties"]["input"].value("type", std::string()) != "string") {
        std::cerr << "Responses custom/freeform bridge did not expose direct string input\\n";
        return 1;
    }

''',
    '''    if (!converted_custom_tool.contains("tools") || converted_custom_tool.at("tools").size() != 1 ||
        converted_custom_tool.at("tools")[0]["function"]["parameters"]["properties"]["input"].value("type", std::string()) != "string") {
        std::cerr << "Responses custom/freeform bridge did not expose direct string input\\n";
        return 1;
    }
    const std::string custom_tool_description =
        converted_custom_tool.at("tools")[0]["function"].value("description", std::string());
    if (custom_tool_description.find("@@ -1,3 +1,4 @@") == std::string::npos ||
        custom_tool_description.find("@@ <context>") == std::string::npos ||
        custom_tool_description.find("+*** literal content") == std::string::npos) {
        std::cerr << "Responses apply_patch bridge did not expose Codex patch-format guidance\\n";
        return 1;
    }

''',
)
