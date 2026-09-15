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
    '''        if (custom_tool.at("name").get<std::string>() == "apply_patch") {
            description +=
                "\\n\\nCodex apply_patch syntax: use `*** Begin Patch`, `*** Update File: <path>`, and `*** End Patch`; "
                "do not emit standard unified-diff range headers such as `@@ -1,3 +1,4 @@`. "
                "For update hunks, the first hunk may begin directly with diff lines, and `@@` or `@@ <context>` may be used as Codex context markers when needed. "
                "Every file-content line in an update hunk must start with a space for context, `+` for an added line, or `-` for a removed line. "
                "A literal file-content line beginning with `***` must still carry its diff prefix, for example `+*** literal content` when inserting it.";
        }
''',
    '''        if (custom_tool.at("name").get<std::string>() == "apply_patch") {
            description +=
                "\\n\\nCodex apply_patch syntax: use `*** Begin Patch`, `*** Update File: <path>`, and `*** End Patch`. "
                "IMPORTANT: `@@` here is NOT a standard unified-diff line-range header. Never emit range headers such as `@@ -10,4 +10,5 @@`. "
                "The form `@@ <context>` means: find the literal source line `<context>` in the target file and use that existing line as an anchor for the following change. "
                "The anchor text must actually exist in the file, and there is no closing `@@`; write `@@ fn example()` rather than `@@ fn example() @@`. "
                "Use bare `@@` to start another chunk when no literal anchor is needed. The first update chunk may omit `@@` entirely and begin directly with diff lines. "
                "Every file-content line in an update hunk must start with a space for context, `+` for an added line, or `-` for a removed line. "
                "A literal file-content line beginning with `***` must still carry its diff prefix, for example `+*** literal content` when inserting it. "
                "Example: `*** Begin Patch\\n*** Update File: src/foo.rs\\n@@ fn calculate()\\n     let old = 1;\\n-    return old;\\n+    return old + 1;\\n*** End Patch`. "
                "In that example, `fn calculate()` must be an actual line in `src/foo.rs`.";
        }
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    if (custom_tool_description.find("@@ -1,3 +1,4 @@") == std::string::npos ||
        custom_tool_description.find("@@ <context>") == std::string::npos ||
        custom_tool_description.find("+*** literal content") == std::string::npos) {
        std::cerr << "Responses apply_patch bridge did not expose Codex patch-format guidance\\n";
        return 1;
    }
''',
    '''    if (custom_tool_description.find("@@ -10,4 +10,5 @@") == std::string::npos ||
        custom_tool_description.find("literal source line") == std::string::npos ||
        custom_tool_description.find("there is no closing `@@`") == std::string::npos ||
        custom_tool_description.find("@@ fn calculate()") == std::string::npos ||
        custom_tool_description.find("must be an actual line") == std::string::npos ||
        custom_tool_description.find("+*** literal content") == std::string::npos) {
        std::cerr << "Responses apply_patch bridge did not expose Codex patch-format guidance\\n";
        return 1;
    }
''',
)
