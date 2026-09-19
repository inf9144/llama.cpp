#include "server-chat.h"
#include "server-task.h"

#include <iostream>
#include <string>

static server_task_result_cmpl_final make_response_result(const std::string & content) {
    server_task_result_cmpl_final result {};
    result.oaicompat_model = "test-model";
    result.oai_resp_id = "resp_test";
    result.oai_resp_message_id = "msg_test";
    result.n_prompt_tokens = 0;
    result.n_prompt_tokens_cache = 0;
    result.n_decoded = 0;
    result.oaicompat_msg.role = "assistant";
    result.oaicompat_msg.content = content;
    return result;
}

static std::string response_message_phase(const json & response) {
    if (!response.contains("output") || !response.at("output").is_array()) {
        return {};
    }

    for (const auto & item : response.at("output")) {
        if (item.value("type", std::string()) == "message") {
            return item.value("phase", std::string());
        }
    }

    return {};
}

static std::string streamed_message_phase(const json & events) {
    for (const auto & event : events) {
        if (event.value("event", std::string()) != "response.output_item.done" ||
            !event.contains("data") || !event.at("data").contains("item")) {
            continue;
        }

        const auto & item = event.at("data").at("item");
        if (item.value("type", std::string()) == "message") {
            return item.value("phase", std::string());
        }
    }

    return {};
}

int main() {
    // Codex sends automatic thread-title requests as Responses structured
    // output. The bridge must preserve the schema for llama.cpp's grammar.
    const json title_schema = {
        {"type", "object"},
        {"properties", {
            {"title", {
                {"type", "string"},
                {"minLength", 1},
                {"maxLength", 36},
            }},
        }},
        {"required", json::array({"title"})},
        {"additionalProperties", false},
    };

    const json title_request = {
        {"input", "Generate a short thread title"},
        {"text", {
            {"format", {
                {"type", "json_schema"},
                {"name", "codex_output_schema"},
                {"strict", true},
                {"schema", title_schema},
            }},
        }},
    };

    const json converted_title =
        server_chat_convert_responses_to_chatcmpl(title_request);

    if (!converted_title.contains("json_schema") ||
        converted_title.at("json_schema") != title_schema ||
        converted_title.contains("text") ||
        converted_title.at("messages").at(0).at("role") != "user") {
        std::cerr << "Responses thread-title JSON schema was not converted correctly\n";
        return 1;
    }

    // The Codex tool router rejects `justification` unless the call also requests
    // unsandboxed execution. The Responses egress must normalize that combination
    // away before the client sees the function_call item.
    auto build_function_call = [](const std::string & name, const std::string & arguments) {
        common_chat_tool_call tool_call;
        tool_call.name      = name;
        tool_call.arguments = arguments;
        tool_call.id        = "test";
        return server_task_build_response_function_call(tool_call, "completed");
    };

    const json stray_justification = build_function_call(
        "shell",
        "{\"command\":\"cat /tmp/x\",\"justification\":\"read the file\"}");
    if (!stray_justification.at("arguments").is_string() ||
        json::parse(stray_justification.at("arguments")).contains("justification")) {
        std::cerr << "Responses egress did not drop a stray justification field\n";
        return 1;
    }

    const json escalated_justification = build_function_call(
        "shell",
        "{\"command\":\"rm -rf /tmp/x\",\"justification\":\"delete the file\",\"sandbox_permissions\":\"require_escalated\"}");
    if (!json::parse(escalated_justification.at("arguments")).contains("justification")) {
        std::cerr << "Responses egress dropped a justification that belongs to an escalated call\n";
        return 1;
    }

    const json plain_arguments = build_function_call("shell", "{\"command\":\"ls\"}");
    if (plain_arguments.at("arguments") != "{\"command\":\"ls\"}") {
        std::cerr << "Responses egress modified arguments without a justification field\n";
        return 1;
    }

    const json non_json_arguments = build_function_call("shell", "not-json");
    if (non_json_arguments.at("arguments") != "not-json") {
        std::cerr << "Responses egress corrupted non-JSON arguments\n";
        return 1;
    }

    auto final_result = make_response_result("Done.");
    if (response_message_phase(final_result.to_json_oaicompat_resp()) != "final_answer") {
        std::cerr << "Responses message without tool calls was not final_answer\n";
        return 1;
    }
    if (streamed_message_phase(final_result.to_json_oaicompat_resp_stream()) != "final_answer") {
        std::cerr << "Streaming Responses message without tool calls was not final_answer\n";
        return 1;
    }

    auto commentary_result = make_response_result("I will inspect the repository.");
    commentary_result.oaicompat_msg.tool_calls.push_back({
        "shell_command",
        R"({"command":"pwd"})",
        "tool_1",
    });

    if (response_message_phase(commentary_result.to_json_oaicompat_resp()) != "commentary") {
        std::cerr << "Responses message with tool calls was not commentary\n";
        return 1;
    }
    if (streamed_message_phase(commentary_result.to_json_oaicompat_resp_stream()) != "commentary") {
        std::cerr << "Streaming Responses message with tool calls was not commentary\n";
        return 1;
    }

    server_task_result_cmpl_partial reasoning_chunk {};
    reasoning_chunk.is_updated = true;
    reasoning_chunk.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
    reasoning_chunk.oai_resp_reasoning_id = "rs_test";
    common_chat_msg_diff reasoning_diff;
    reasoning_diff.reasoning_content_delta = "Thinking...";
    reasoning_chunk.oaicompat_msg_diffs.push_back(reasoning_diff);

    const auto reasoning_events = reasoning_chunk.to_json_oaicompat_resp();
    bool found_reasoning_delta = false;
    for (const auto & event : reasoning_events) {
        if (event.value("event", std::string()) != "response.reasoning_text.delta") {
            continue;
        }
        found_reasoning_delta = true;
        const auto & data = event.at("data");
        if (data.value("item_id", std::string()) != "rs_test" ||
            data.value("delta", std::string()) != "Thinking..." ||
            data.value("content_index", -1) != 0) {
            std::cerr << "Streaming Responses reasoning delta metadata was incorrect\n";
            return 1;
        }
    }
    if (!found_reasoning_delta) {
        std::cerr << "Streaming Responses reasoning delta was not emitted\n";
        return 1;
    }

    std::cout << "Responses assistant message phase and reasoning stream tests passed\n";
    return 0;
}
