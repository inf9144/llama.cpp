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
    '''                "Use bare `@@` to start another chunk when no literal anchor is needed. The first update chunk may omit `@@` entirely and begin directly with diff lines. "
                "Every file-content line in an update hunk must start with a space for context, `+` for an added line, or `-` for a removed line. "
''',
    '''                "Use bare `@@` to start another chunk when no literal anchor is needed. The first update chunk may omit `@@` entirely and begin directly with diff lines. "
                "For a pure append to the end of an existing file, use an Update File chunk containing only `+` lines with no context and no `@@`; an add-only chunk with no old/context lines is inserted at EOF. Prefer this over copying a long final-line anchor or generating helper scripts just to preserve anchor bytes. "
                "Every file-content line in an update hunk must start with a space for context, `+` for an added line, or `-` for a removed line. "
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''        custom_tool_description.find("@@ fn calculate()") == std::string::npos ||
        custom_tool_description.find("must be an actual line") == std::string::npos ||
        custom_tool_description.find("+*** literal content") == std::string::npos) {
''',
    '''        custom_tool_description.find("@@ fn calculate()") == std::string::npos ||
        custom_tool_description.find("must be an actual line") == std::string::npos ||
        custom_tool_description.find("pure append") == std::string::npos ||
        custom_tool_description.find("inserted at EOF") == std::string::npos ||
        custom_tool_description.find("+*** literal content") == std::string::npos) {
''',
)
