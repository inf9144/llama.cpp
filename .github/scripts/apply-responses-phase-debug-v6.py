from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


server_task_path = Path("tools/server/server-task.cpp")
server_task = server_task_path.read_text()

# v5 initially emitted every buffered *.added/*.delta prelude before any *.done
# event. Keep only the reasoning prelude there; message/tool preludes are moved
# beside their matching done blocks below so Responses item ordering stays
# reasoning -> message -> tool call.
start = server_task.find('''    if (buffered_responses && oaicompat_msg.reasoning_content != "") {
''')
end = server_task.find('''    if (oaicompat_msg.reasoning_content != "") {
''', start + 1)
if start < 0 or end < 0:
    raise SystemExit("buffered prelude block markers not found")
reasoning_only = '''    if (buffered_responses && oaicompat_msg.reasoning_content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_item.added"},
            {"data", json {
                {"type", "response.output_item.added"},
                {"item", json {
                    {"id",                oai_resp_reasoning_id},
                    {"summary",           json::array()},
                    {"type",              "reasoning"},
                    {"content",           json::array()},
                    {"encrypted_content", ""},
                    {"status",            "in_progress"},
                }},
            }},
        });
        server_sent_events.push_back(json {
            {"event", "response.reasoning_text.delta"},
            {"data", json {
                {"type",    "response.reasoning_text.delta"},
                {"delta",   oaicompat_msg.reasoning_content},
                {"item_id", oai_resp_reasoning_id},
            }},
        });
    }

'''
server_task = server_task[:start] + reasoning_only + server_task[end:]

server_task = replace_once(
    server_task,
'''        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
''',
'''        output.push_back(output_item);
    }

    if (buffered_responses && oaicompat_msg.content != "") {
        const std::string resolved_phase =
            server_task_response_message_phase_debug(oaicompat_msg, "response.output_item.added.buffered");
        server_sent_events.push_back(json {
            {"event", "response.output_item.added"},
            {"data", json {
                {"type", "response.output_item.added"},
                {"item", json {
                    {"content", json::array()},
                    {"id",      oai_resp_message_id},
                    {"role",    "assistant"},
                    {"phase",   resolved_phase},
                    {"status",  "in_progress"},
                    {"type",    "message"},
                }},
            }},
        });
        server_sent_events.push_back(json {
            {"event", "response.content_part.added"},
            {"data", json {
                {"type",    "response.content_part.added"},
                {"item_id", oai_resp_message_id},
                {"part", json {
                    {"type", "output_text"},
                    {"text", ""},
                }},
            }},
        });
        server_sent_events.push_back(json {
            {"event", "response.output_text.delta"},
            {"data", json {
                {"type",    "response.output_text.delta"},
                {"item_id", oai_resp_message_id},
                {"delta",   oaicompat_msg.content},
            }},
        });
    }

    if (oaicompat_msg.content != "") {
''',
    "move buffered message prelude after reasoning done",
)

server_task = replace_once(
    server_task,
'''        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const bool is_custom = server_task_is_response_custom_tool(
''',
'''        output.push_back(output_item);
    }

    if (buffered_responses) {
        for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
            const bool is_custom = server_task_is_response_custom_tool(
                generation_params.responses_custom_tools, tool_call);
            const bool is_tool_search = server_task_is_response_tool_search(
                generation_params.responses_tool_search, tool_call);
            if (is_custom || is_tool_search) {
                continue;
            }
            json added_item = server_task_build_response_function_call(tool_call, "in_progress");
            added_item["arguments"] = "";
            server_sent_events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type", "response.output_item.added"},
                    {"item", std::move(added_item)},
                }},
            });
            server_sent_events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   tool_call.arguments},
                    {"item_id", "fc_" + tool_call.id},
                }},
            });
        }
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const bool is_custom = server_task_is_response_custom_tool(
''',
    "move buffered tool prelude after message done",
)

server_task_path.write_text(server_task)

# Strengthen the focused regression so mere event presence is insufficient: the
# message item must not start until the reasoning item is done.
test_path = Path("tests/test-qwen38-codex-template.cpp")
test = test_path.read_text()
test = replace_once(
    test,
'''    bool saw_final_added = false;
    bool saw_final_delta = false;
    bool saw_final_done = false;
    bool saw_final_completed = false;
    for (const auto & event : buffered_final_result.to_json_oaicompat_resp_stream()) {
        const std::string type = event.value("event", std::string());
''',
'''    bool saw_final_added = false;
    bool saw_final_delta = false;
    bool saw_final_done = false;
    bool saw_final_completed = false;
    bool saw_reasoning_done = false;
    bool message_started_before_reasoning_done = false;
    for (const auto & event : buffered_final_result.to_json_oaicompat_resp_stream()) {
        const std::string type = event.value("event", std::string());
        if (type == "response.output_item.done" &&
            event.contains("data") && event.at("data").contains("item") &&
            event.at("data").at("item").value("type", std::string()) == "reasoning") {
            saw_reasoning_done = true;
        }
''',
    "track buffered item ordering",
)
test = replace_once(
    test,
'''            saw_final_added =
                event.at("data").at("item").value("phase", std::string()) == "final_answer";
''',
'''            if (!saw_reasoning_done) {
                message_started_before_reasoning_done = true;
            }
            saw_final_added =
                event.at("data").at("item").value("phase", std::string()) == "final_answer";
''',
    "assert message starts after reasoning",
)
test = replace_once(
    test,
'''    if (!saw_final_added || !saw_final_delta || !saw_final_done || !saw_final_completed) {
        std::cerr << "Buffered Responses final stream lost phase/event ordering\\n";
''',
'''    if (!saw_final_added || !saw_final_delta || !saw_final_done || !saw_final_completed ||
        !saw_reasoning_done || message_started_before_reasoning_done) {
        std::cerr << "Buffered Responses final stream lost phase/event ordering\\n";
''',
    "enforce buffered item ordering",
)
test_path.write_text(test)
