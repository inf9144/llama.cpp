from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    p.write_text(text.replace(old, new, 1))


# Fix the no-varargs SRV_WRN invocation: SRV_WRN expands with a mandatory
# trailing __VA_ARGS__ in the current server logging macro.
replace_once(
    "tools/server/server-chat.cpp",
    '                    SRV_WRN("unsupported server-executed Responses tool_search skipped\\n");\n',
    '                    SRV_WRN("%s\\n", "unsupported server-executed Responses tool_search skipped");\n',
)

# The final streaming path only needs the generic response-item builder; unlike
# custom tools, tool_search has no input-delta event, so no local flag is needed.
replace_once(
    "tools/server/server-task.cpp",
    '''        const bool is_tool_search = server_task_is_response_tool_search(\n            generation_params.responses_tool_search, tool_call);\n\n        if (is_custom) {\n''',
    '''        if (is_custom) {\n''',
)

# Exercise the real Responses egress, not only converter/parser plumbing.
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '#include "server-chat.h"\n',
    '#include "server-chat.h"\n#include "server-task.h"\n',
)

marker = '    std::cout << "Qwen3.8 Codex user text, generic string framing, custom framing, and compaction tests passed\\n";\n'
addition = r'''    // Verify that an internally parsed tool_search function call is emitted
    // using Codex's dedicated Responses item instead of a normal function_call.
    server_task_result_cmpl_final tool_search_result;
    tool_search_result.oaicompat_model = "test-model";
    tool_search_result.oai_resp_id = "resp_tool_search_test";
    tool_search_result.generation_params.responses_tool_search = true;
    tool_search_result.oaicompat_msg.role = "assistant";
    tool_search_result.oaicompat_msg.tool_calls.push_back({
        "tool_search",
        nlohmann::ordered_json({{"query", "calendar"}, {"limit", 8}}).dump(),
        "search_generated",
    });

    const auto tool_search_response = tool_search_result.to_json_oaicompat_resp();
    if (!tool_search_response.contains("output") || tool_search_response.at("output").size() != 1) {
        std::cerr << "Responses tool_search egress did not produce exactly one output item\n";
        return 1;
    }
    const auto & tool_search_item = tool_search_response.at("output")[0];
    if (tool_search_item.value("type", std::string()) != "tool_search_call" ||
        tool_search_item.value("execution", std::string()) != "client" ||
        tool_search_item.value("call_id", std::string()) != "call_search_generated" ||
        !tool_search_item.contains("arguments") || !tool_search_item.at("arguments").is_object() ||
        tool_search_item.at("arguments").value("query", std::string()) != "calendar" ||
        tool_search_item.at("arguments").value("limit", 0) != 8) {
        std::cerr << "Responses tool_search egress emitted the wrong wire shape\n";
        return 1;
    }

    const auto tool_search_stream = tool_search_result.to_json_oaicompat_resp_stream();
    bool saw_tool_search_done = false;
    bool saw_function_argument_delta = false;
    for (const auto & event : tool_search_stream) {
        if (event.value("event", std::string()) == "response.function_call_arguments.delta") {
            saw_function_argument_delta = true;
        }
        if (event.value("event", std::string()) == "response.output_item.done" &&
            event.contains("data") && event.at("data").contains("item") &&
            event.at("data").at("item").value("type", std::string()) == "tool_search_call") {
            saw_tool_search_done = true;
        }
    }
    if (!saw_tool_search_done || saw_function_argument_delta) {
        std::cerr << "Responses tool_search streaming used normal function-call streaming semantics\n";
        return 1;
    }

'''
replace_once("tests/test-qwen38-codex-template.cpp", marker, addition + marker)

print("tool_search compile fixes and egress tests applied")
