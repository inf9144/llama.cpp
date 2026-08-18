from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    p.write_text(text.replace(old, new, 1))


# Accumulate a unique parser-visible tool set across multiple historical
# tool_search_output items. Each chronological output message still keeps its
# full result; this only prevents duplicate PEG/parser definitions.
replace_once(
    "tools/server/server-chat.cpp",
    '''    std::vector<json> responses_deferred_tools;\n    std::set<std::string> responses_custom_tools;\n''',
    '''    std::vector<json> responses_deferred_tools;\n    std::set<std::string> responses_deferred_tool_names;\n    std::set<std::string> responses_custom_tools;\n''',
)
replace_once(
    "tools/server/server-chat.cpp",
    '''                responses_deferred_tools.insert(\n                    responses_deferred_tools.end(), discovered_tools.begin(), discovered_tools.end());\n\n                chatcmpl_messages.push_back(json {\n''',
    '''                for (const json & discovered_tool : discovered_tools) {\n                    if (!discovered_tool.contains("function") ||\n                        !discovered_tool.at("function").is_object() ||\n                        !exists_and_is_string(discovered_tool.at("function"), "name")) {\n                        continue;\n                    }\n                    const std::string discovered_name =\n                        discovered_tool.at("function").at("name").get<std::string>();\n                    if (responses_deferred_tool_names.insert(discovered_name).second) {\n                        responses_deferred_tools.push_back(discovered_tool);\n                    }\n                }\n\n                chatcmpl_messages.push_back(json {\n''',
)

# Deferred tool specs are consumed entirely while formatting the chat request;
# they do not need to survive into the inference task parameter object.
replace_once(
    "tools/server/server-common.cpp",
    '''    auto tools = json_value(body, "tools", json());\n    auto deferred_tools = json_value(body, "__llamacpp_responses_deferred_tools", json());\n    auto has_tools = tools.is_array() && !tools.empty();\n''',
    '''    auto tools = json_value(body, "tools", json());\n    auto deferred_tools = json_value(body, "__llamacpp_responses_deferred_tools", json());\n    body.erase("__llamacpp_responses_deferred_tools");\n    auto has_tools = (tools.is_array() && !tools.empty()) ||\n        (deferred_tools.is_array() && !deferred_tools.empty());\n''',
)

# Regression: repeated searches may rediscover the same tool, but the internal
# parser tool set must contain it only once while both output messages remain in
# chronological history.
marker = '    // Verify that an internally parsed tool_search function call is emitted\n'
addition = r'''    auto repeated_search_request = followup_request;
    auto repeated_call = repeated_search_request.at("input")[1];
    repeated_call["call_id"] = "search-2";
    auto repeated_output = repeated_search_request.at("input")[2];
    repeated_output["call_id"] = "search-2";
    repeated_search_request["input"].push_back(repeated_call);
    repeated_search_request["input"].push_back(repeated_output);
    const auto converted_repeated_search =
        server_chat_convert_responses_to_chatcmpl(repeated_search_request);
    if (!converted_repeated_search.contains("__llamacpp_responses_deferred_tools") ||
        converted_repeated_search.at("__llamacpp_responses_deferred_tools").size() != 1) {
        std::cerr << "Repeated tool_search results duplicated the parser-visible tool set\n";
        return 1;
    }
    size_t repeated_tool_results = 0;
    for (const auto & msg : converted_repeated_search.at("messages")) {
        if (msg.value("role", std::string()) == "tool" &&
            (msg.value("tool_call_id", std::string()) == "search-1" ||
             msg.value("tool_call_id", std::string()) == "search-2")) {
            ++repeated_tool_results;
        }
    }
    if (repeated_tool_results != 2) {
        std::cerr << "Repeated tool_search outputs were not preserved chronologically\n";
        return 1;
    }

'''
replace_once("tests/test-qwen38-codex-template.cpp", marker, addition + marker)

print("tool_search robustness hardening applied")
