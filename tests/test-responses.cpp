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

// Codex replays every Responses output item in subsequent sampling requests.
// Distinct reasoning-only responses must survive the Responses -> Chat bridge.
static bool test_responses_reasoning_replay() {
    auto reasoning_item = [](const std::string & text) {
        return json {
            {"type", "reasoning"},
            {"summary", json::array()},
            {"content", json::array({json {
                {"type", "reasoning_text"},
                {"text", text},
            }})},
        };
    };
    auto tool_call = [](const std::string & id) {
        return json {
            {"type", "function_call"},
            {"call_id", id},
            {"name", "exec_command"},
            {"arguments", R"({"cmd":"pwd"})"},
        };
    };
    auto tool_result = [](const std::string & id) {
        return json {
            {"type", "function_call_output"},
            {"call_id", id},
            {"output", "/work"},
        };
    };
    auto start_input = []() {
        return json::array({json {
            {"role", "user"},
            {"content", "Complete the task."},
        }});
    };
    auto replay = [](const json & input) -> json {
        return server_chat_convert_responses_to_chatcmpl(json {
            {"input", input},
        }).at("messages");
    };
    auto failed = [](const char * scenario) {
        std::cerr << "Responses reasoning replay failed: " << scenario << "\n";
        return false;
    };

    // 1. Reasoning A -> Reasoning B: neither block may be replaced.
    auto input = start_input();
    input.push_back(reasoning_item("Reasoning A"));
    input.push_back(reasoning_item("Reasoning B"));
    const json consecutive = replay(input);
    if (consecutive.size() != 2 ||
        consecutive[1].at("role") != "assistant" ||
        consecutive[1].at("reasoning_content") != "Reasoning A\n\nReasoning B" ||
        !consecutive[1].at("content").empty()) {
        return failed("A -> B");
    }

    // 2. Reasoning A -> Reasoning B + Tool-Call: the call belongs to
    // the same assistant message as both reasoning segments.
    input.push_back(tool_call("call_b"));
    const json with_tool = replay(input);
    if (with_tool.size() != 2 ||
        with_tool[1].at("reasoning_content") != "Reasoning A\n\nReasoning B" ||
        with_tool[1].at("tool_calls").size() != 1 ||
        with_tool[1].at("tool_calls")[0].at("id") != "call_b" ||
        with_tool[1].at("tool_calls")[0].at("function").at("name") != "exec_command") {
        return failed("A -> B + tool call");
    }

    // 3. A -> B + Call -> Result -> C + Call: a tool result creates
    // a hard boundary. C must not be attached to A/B.
    input.push_back(tool_result("call_b"));
    input.push_back(reasoning_item("Reasoning C"));
    input.push_back(tool_call("call_c"));
    const json separated = replay(input);
    if (separated.size() != 4 ||
        separated[1].at("reasoning_content") != "Reasoning A\n\nReasoning B" ||
        separated[1].at("tool_calls")[0].at("id") != "call_b" ||
        separated[2].at("role") != "tool" ||
        separated[2].at("tool_call_id") != "call_b" ||
        separated[2].at("content") != "/work" ||
        separated[3].at("role") != "assistant" ||
        separated[3].at("reasoning_content") != "Reasoning C" ||
        separated[3].at("tool_calls")[0].at("id") != "call_c") {
        return failed("A -> B + tool -> tool result -> C + tool");
    }

    // 4. Long autonomous reasoning-only continuation chain: verify
    // every block remains in order, with no intervening user messages.
    json long_input = start_input();
    std::string expected_reasoning;
    for (int i = 0; i < 256; ++i) {
        const std::string segment = "Reasoning segment " + std::to_string(i);
        long_input.push_back(reasoning_item(segment));
        if (!expected_reasoning.empty()) {
            expected_reasoning += "\n\n";
        }
        expected_reasoning += segment;
    }
    long_input.push_back(tool_call("call_long"));
    const json long_replay = replay(long_input);
    if (long_replay.size() != 2 ||
        long_replay[1].at("reasoning_content") != expected_reasoning ||
        long_replay[1].at("tool_calls").size() != 1 ||
        long_replay[1].at("tool_calls")[0].at("id") != "call_long") {
        return failed("256 reasoning-only continuations");
    }

    // Existing assistant output text still merges with the reasoning.
    json final_input = start_input();
    final_input.push_back(reasoning_item("Reasoning A"));
    final_input.push_back(reasoning_item("Reasoning B"));
    final_input.push_back(json {
        {"type", "message"},
        {"role", "assistant"},
        {"content", json::array({json {
            {"type", "output_text"},
            {"text", "Final answer."},
        }})},
    });
    const json final_replay = replay(final_input);
    if (final_replay.size() != 2 ||
        final_replay[1].at("reasoning_content") != "Reasoning A\n\nReasoning B" ||
        final_replay[1].at("content").size() != 1 ||
        final_replay[1].at("content")[0].at("text") != "Final answer.") {
        return failed("reasoning -> final answer");
    }

    // Existing compaction behavior also remains a message boundary.
    json compact_input = start_input();
    compact_input.push_back(reasoning_item("Reasoning before compaction"));
    compact_input.push_back(json {
        {"type", "compaction"},
        {"encrypted_content", "llama.cpp.compaction.v1\nCheckpoint"},
    });
    compact_input.push_back(reasoning_item("Reasoning after compaction"));
    compact_input.push_back(tool_call("call_after_compaction"));
    const json compact_replay = replay(compact_input);
    if (compact_replay.size() != 4 ||
        compact_replay[1].at("reasoning_content") != "Reasoning before compaction" ||
        compact_replay[2].at("role") != "user" ||
        compact_replay[2].at("content") != "<context_compaction>\nCheckpoint\n</context_compaction>" ||
        compact_replay[3].at("reasoning_content") != "Reasoning after compaction" ||
        compact_replay[3].at("tool_calls")[0].at("id") != "call_after_compaction") {
        return failed("reasoning and compaction boundary");
    }

    return true;
}

int main() {
    if (!test_responses_reasoning_replay()) {
        return 1;
    }

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

    // The Codex tool router rejects `justification` on exec_command unless the call
    // also requests unsandboxed execution. The Responses egress must normalize that
    // combination away before the client sees the function_call item, and only for
    // the exec_command tool itself.
    auto build_function_call = [](const std::string & name, const std::string & arguments) {
        common_chat_tool_call tool_call;
        tool_call.name      = name;
        tool_call.arguments = arguments;
        tool_call.id        = "test";
        return server_task_build_response_function_call(tool_call, "completed");
    };

    const json stray_justification = build_function_call(
        "exec_command",
        "{\"command\":\"cat /tmp/x\",\"justification\":\"read the file\"}");
    if (!stray_justification.at("arguments").is_string() ||
        json::parse(stray_justification.at("arguments")).contains("justification")) {
        std::cerr << "Responses egress did not drop a stray exec_command justification\n";
        return 1;
    }

    const json escalated_justification = build_function_call(
        "exec_command",
        "{\"command\":\"rm -rf /tmp/x\",\"justification\":\"delete the file\",\"sandbox_permissions\":\"require_escalated\"}");
    if (!json::parse(escalated_justification.at("arguments")).contains("justification")) {
        std::cerr << "Responses egress dropped a justification that belongs to an escalated call\n";
        return 1;
    }

    const json plain_arguments = build_function_call("exec_command", "{\"command\":\"ls\"}");
    if (plain_arguments.at("arguments") != "{\"command\":\"ls\"}") {
        std::cerr << "Responses egress modified arguments without a justification field\n";
        return 1;
    }

    const json non_json_arguments = build_function_call("exec_command", "not-json");
    if (non_json_arguments.at("arguments") != "not-json") {
        std::cerr << "Responses egress corrupted non-JSON arguments\n";
        return 1;
    }

    // Other tools must pass through untouched, even with the same field combination.
    const json other_tool_justification = build_function_call(
        "apply_patch",
        "{\"input\":\"*** Begin Patch\",\"justification\":\"edit the file\"}");
    if (!json::parse(other_tool_justification.at("arguments")).contains("justification")) {
        std::cerr << "Responses egress normalized a justification on a non-exec_command tool\n";
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

    // A reasoning-only EOS is a completed Responses request but not the end
    // of the Codex agent turn. Verify both streaming and non-streaming egress.
    auto check_end_turn = [](server_task_result_cmpl_final & result,
                             bool expected_continuation,
                             const std::string & case_name) {
        const json response = result.to_json_oaicompat_resp();
        const json events = result.to_json_oaicompat_resp_stream();
        const json & streamed_response = events.back().at("data").at("response");

        const bool nonstream_continue = response.value("end_turn", true) == false;
        const bool stream_continue = streamed_response.value("end_turn", true) == false;
        if (nonstream_continue != expected_continuation ||
            stream_continue != expected_continuation) {
            std::cerr << "Unexpected Codex end_turn for " << case_name
                      << " (non-stream=" << nonstream_continue
                      << ", stream=" << stream_continue << ")\n";
            return false;
        }

        if (expected_continuation) {
            if (response.at("status") != "completed" ||
                streamed_response.at("status") != "completed" ||
                response.at("output").size() != 1 ||
                response.at("output")[0].value("type", std::string()) != "reasoning" ||
                streamed_response.at("output").size() != 1 ||
                streamed_response.at("output")[0].value("type", std::string()) != "reasoning") {
                std::cerr << "Reasoning-only EOS lost its completed reasoning output\n";
                return false;
            }
        }
        return true;
    };

    auto reasoning_only = make_response_result("");
    reasoning_only.stop = STOP_TYPE_EOS;
    reasoning_only.oaicompat_msg.reasoning_content = "Need to keep working.";
    if (!check_end_turn(reasoning_only, true, "reasoning-only EOS")) {
        return 1;
    }

    auto answered = make_response_result("Done.");
    answered.stop = STOP_TYPE_EOS;
    answered.oaicompat_msg.reasoning_content = "Finished reasoning.";
    if (!check_end_turn(answered, false, "answered EOS")) {
        return 1;
    }

    auto tool_response = make_response_result("");
    tool_response.stop = STOP_TYPE_EOS;
    tool_response.oaicompat_msg.reasoning_content = "A tool is needed.";
    tool_response.oaicompat_msg.tool_calls.push_back({
        "shell_command", R"({"command":"pwd"})", "tool_2",
    });
    if (!check_end_turn(tool_response, false, "tool-call EOS")) {
        return 1;
    }

    auto no_reasoning = make_response_result("");
    no_reasoning.stop = STOP_TYPE_EOS;
    if (!check_end_turn(no_reasoning, false, "empty EOS without reasoning")) {
        return 1;
    }

    auto hit_limit = make_response_result("");
    hit_limit.stop = STOP_TYPE_LIMIT;
    hit_limit.oaicompat_msg.reasoning_content = "Thinking was interrupted.";
    if (!check_end_turn(hit_limit, false, "reasoning-only limit")) {
        return 1;
    }

    auto compaction = make_response_result("");
    compaction.stop = STOP_TYPE_EOS;
    compaction.generation_params.responses_compaction = true;
    compaction.oaicompat_msg.reasoning_content = "Summarizing.";
    compaction.oai_resp_compaction_content = "Current goal and progress.";
    if (!check_end_turn(compaction, false, "compaction EOS")) {
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
