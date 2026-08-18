from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one marker in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1))


replace_once(
    "tools/server/server-task.cpp",
    '''static json server_task_build_response_tool_call(
        const common_chat_tool_call & tool_call,
        const std::string & status,
        const std::unordered_set<std::string> & custom_tools,
        bool responses_tool_search) {
    if (server_task_is_response_tool_search(responses_tool_search, tool_call)) {
        return server_task_build_response_tool_search_call(tool_call, status);
    }
    return server_task_is_response_custom_tool(custom_tools, tool_call)
        ? server_task_build_response_custom_tool_call(tool_call, status)
        : server_task_build_response_function_call(tool_call, status);
}

//
// task_params
//
''',
    '''static json server_task_build_response_tool_call(
        const common_chat_tool_call & tool_call,
        const std::string & status,
        const std::unordered_set<std::string> & custom_tools,
        bool responses_tool_search) {
    if (server_task_is_response_tool_search(responses_tool_search, tool_call)) {
        return server_task_build_response_tool_search_call(tool_call, status);
    }
    return server_task_is_response_custom_tool(custom_tools, tool_call)
        ? server_task_build_response_custom_tool_call(tool_call, status)
        : server_task_build_response_function_call(tool_call, status);
}

// Qwen's parser exposes assistant text and tool calls together only once the
// response item is complete. Text accompanying a tool call is mid-turn
// commentary; text without a tool call is terminal for this sampling response.
static const char * server_task_response_message_phase(const common_chat_msg & msg) {
    return msg.tool_calls.empty() ? "final_answer" : "commentary";
}

//
// task_params
//
''',
)

replace_once(
    "tools/server/server-task.cpp",
    '''            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
''',
    '''            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"phase",  server_task_response_message_phase(msg)},
            {"status", "completed"},
            {"type",   "message"},
''',
)

replace_once(
    "tools/server/server-task.cpp",
    '''            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
''',
    '''            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"},
            {"phase",   server_task_response_message_phase(oaicompat_msg)}
''',
)

replace_once(
    "tools/server/server-chat.cpp",
    '''                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
''',
    '''                // Responses output-message phase is client-facing lifecycle metadata.
                // The Qwen chat template gets the same semantics from chronological
                // assistant text + tool calls, so do not leak it into Chat Completions.
                item.erase("phase");

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
''',
)

replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    if (!saw_tool_search_done || saw_function_argument_delta) {
        std::cerr << "Responses tool_search streaming used normal function-call streaming semantics\\n";
        return 1;
    }

    std::cout << "Qwen3.8 Codex user text, generic string framing, custom framing, and compaction tests passed\\n";
''',
    '''    if (!saw_tool_search_done || saw_function_argument_delta) {
        std::cerr << "Responses tool_search streaming used normal function-call streaming semantics\\n";
        return 1;
    }

    // Codex 0.147 consumes output-message phase as lifecycle metadata. Once
    // recorded in history it comes back on the next Responses request, where
    // the bridge must consume it rather than forwarding it to Chat Completions.
    const nlohmann::ordered_json phase_history_request = {
        {"model", "test-model"},
        {"input", nlohmann::ordered_json::array({
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
    };
    const auto converted_phase_history = server_chat_convert_responses_to_chatcmpl(phase_history_request);
    if (!converted_phase_history.contains("messages") ||
        converted_phase_history.at("messages").size() != 1 ||
        converted_phase_history.at("messages")[0].contains("phase") ||
        converted_phase_history.at("messages")[0].value("role", std::string()) != "assistant") {
        std::cerr << "Responses message phase leaked into Chat Completions history\\n";
        return 1;
    }

    server_task_result_cmpl_final final_phase_result;
    final_phase_result.oaicompat_model = "test-model";
    final_phase_result.oai_resp_id = "resp_final_phase_test";
    final_phase_result.oai_resp_message_id = "msg_final_phase_test";
    final_phase_result.n_prompt_tokens = 0;
    final_phase_result.n_prompt_tokens_cache = 0;
    final_phase_result.n_decoded = 0;
    final_phase_result.oaicompat_msg.role = "assistant";
    final_phase_result.oaicompat_msg.content = "Done.";

    const auto final_phase_response = final_phase_result.to_json_oaicompat_resp();
    if (!final_phase_response.contains("output") || final_phase_response.at("output").size() != 1 ||
        final_phase_response.at("output")[0].value("type", std::string()) != "message" ||
        final_phase_response.at("output")[0].value("phase", std::string()) != "final_answer") {
        std::cerr << "Terminal Responses text was not classified as final_answer\\n";
        return 1;
    }

    bool saw_final_phase_done = false;
    bool saw_final_phase_completed = false;
    for (const auto & event : final_phase_result.to_json_oaicompat_resp_stream()) {
        if (event.value("event", std::string()) == "response.output_item.done" &&
            event.contains("data") && event.at("data").contains("item") &&
            event.at("data").at("item").value("type", std::string()) == "message") {
            saw_final_phase_done =
                event.at("data").at("item").value("phase", std::string()) == "final_answer";
        }
        if (event.value("event", std::string()) == "response.completed" &&
            event.contains("data") && event.at("data").contains("response") &&
            event.at("data").at("response").contains("output") &&
            !event.at("data").at("response").at("output").empty()) {
            saw_final_phase_completed =
                event.at("data").at("response").at("output")[0].value("phase", std::string()) == "final_answer";
        }
    }
    if (!saw_final_phase_done || !saw_final_phase_completed) {
        std::cerr << "Streaming terminal Responses text lost final_answer phase\\n";
        return 1;
    }

    server_task_result_cmpl_final commentary_phase_result;
    commentary_phase_result.oaicompat_model = "test-model";
    commentary_phase_result.oai_resp_id = "resp_commentary_phase_test";
    commentary_phase_result.oai_resp_message_id = "msg_commentary_phase_test";
    commentary_phase_result.n_prompt_tokens = 0;
    commentary_phase_result.n_prompt_tokens_cache = 0;
    commentary_phase_result.n_decoded = 0;
    commentary_phase_result.oaicompat_msg.role = "assistant";
    commentary_phase_result.oaicompat_msg.content = "I'll inspect it.";
    commentary_phase_result.oaicompat_msg.tool_calls.push_back({
        "shell_command",
        nlohmann::ordered_json({{"command", "pwd"}}).dump(),
        "phase_tool",
    });

    const auto commentary_phase_response = commentary_phase_result.to_json_oaicompat_resp();
    if (!commentary_phase_response.contains("output") || commentary_phase_response.at("output").size() != 2 ||
        commentary_phase_response.at("output")[0].value("type", std::string()) != "message" ||
        commentary_phase_response.at("output")[0].value("phase", std::string()) != "commentary" ||
        commentary_phase_response.at("output")[1].value("type", std::string()) != "function_call") {
        std::cerr << "Responses text preceding a tool call was not classified as commentary\\n";
        return 1;
    }

    bool saw_commentary_phase_done = false;
    bool saw_commentary_phase_completed = false;
    for (const auto & event : commentary_phase_result.to_json_oaicompat_resp_stream()) {
        if (event.value("event", std::string()) == "response.output_item.done" &&
            event.contains("data") && event.at("data").contains("item") &&
            event.at("data").at("item").value("type", std::string()) == "message") {
            saw_commentary_phase_done =
                event.at("data").at("item").value("phase", std::string()) == "commentary";
        }
        if (event.value("event", std::string()) == "response.completed" &&
            event.contains("data") && event.at("data").contains("response") &&
            event.at("data").at("response").contains("output") &&
            !event.at("data").at("response").at("output").empty()) {
            saw_commentary_phase_completed =
                event.at("data").at("response").at("output")[0].value("phase", std::string()) == "commentary";
        }
    }
    if (!saw_commentary_phase_done || !saw_commentary_phase_completed) {
        std::cerr << "Streaming pre-tool Responses text lost commentary phase\\n";
        return 1;
    }

    std::cout << "Qwen3.8 Codex user text, generic string framing, custom framing, compaction, tool search, and Responses phase tests passed\\n";
''',
)
