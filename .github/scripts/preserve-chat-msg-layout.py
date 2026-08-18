from pathlib import Path

path = Path("common/chat.h")
text = path.read_text()
old = '''    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               phase;
    std::string                               tool_name;
    std::string                               tool_call_id;
'''
new = '''    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               tool_name;
    std::string                               tool_call_id;
    // Optional assistant lifecycle metadata used by Responses-capable parsers.
    // Keep this last so existing aggregate initializers retain their field mapping.
    std::string                               phase;
'''
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected exactly one common_chat_msg field block, found {count}")
path.write_text(text.replace(old, new, 1))
