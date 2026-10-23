#include "server-chat.h"
#include "server-task.h"
#include "chat.h"

#include <fstream>
#include <iostream>
#include <iterator>
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

// Integration: feed synthetic generation text through the production session
// parser, then through the Responses finalization path. This proves the updated
// session parser yields the expected end_turn result when the model ends after
// reasoning only, without any model inference.
static bool test_responses_reasoning_parser_integration() {
    const std::string template_path = "models/templates/llama-cpp-qwen-based-codex.jinja";
    std::ifstream file(template_path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open template: " << template_path << "\n";
        return false;
    }
    const std::string template_text = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    auto tmpls = common_chat_templates_ptr(common_chat_templates_init(nullptr, template_text));

    common_chat_tool shell_tool;
    shell_tool.name = "shell_command";
    shell_tool.description = "Runs a shell command";
    shell_tool.parameters = R"({"type":"object","properties":{"command":{"type":"string"},"workdir":{"type":"string"}},"required":["command"]})";

    common_chat_msg system;
    system.role = "system";
    system.content = "Stable Codex instructions.";
    common_chat_msg user;
    user.role = "user";
    user.content = "Initial user request.";

    // feed `generated` through the production session parser, then finalize it as a
    // Responses result with the given stop type and compaction flag
    struct gen_out {
        json response;
        json events;
        common_chat_msg msg;
    };
    auto run_generation = [&](const std::string & generated, stop_type stop, bool compaction) -> gen_out {
        common_chat_templates_inputs inputs;
        inputs.messages = { system, user };
        inputs.tools = { shell_tool };
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = true;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

        common_chat_session session(tmpls.get(), nullptr, inputs);
        session.feed(common_chat_input(generated));
        const common_chat_msg & msg = session.finish();

        server_task_result_cmpl_final result {};
        result.oaicompat_model = "test-model";
        result.oai_resp_id = "resp_test";
        result.stop = stop;
        result.generation_params.responses_compaction = compaction;
        result.oai_resp_compaction_content = compaction ? "Current goal and progress." : "";
        result.oaicompat_msg = msg;

        return gen_out { result.to_json_oaicompat_resp(), result.to_json_oaicompat_resp_stream(), msg };
    };

    auto end_turn_false = [](const json & response, const json & events) {
        const json & streamed = events.back().at("data").at("response");
        return response.value("end_turn", true) == false &&
               streamed.value("end_turn", true) == false;
    };
    auto end_turn_true = [](const json & response, const json & events) {
        const json & streamed = events.back().at("data").at("response");
        return response.value("end_turn", true) == true &&
               streamed.value("end_turn", true) == true;
    };

    // 1. Reasoning-only followed by EOS: a completed request, but Codex must continue
    {
        auto r = run_generation("Need to keep working.\n</think>\n\n", STOP_TYPE_EOS, false);
        if (!end_turn_false(r.response, r.events)) {
            std::cerr << "Integration: reasoning-only EOS did not set end_turn=false\n";
            return false;
        }
        if (r.msg.reasoning_content.find("Need to keep working.") == std::string::npos ||
            !r.msg.content.empty() || !r.msg.tool_calls.empty()) {
            std::cerr << "Integration: reasoning-only EOS parser produced unexpected content\n";
            return false;
        }
        if (r.response.at("status") != "completed" ||
            r.response.at("output").size() != 1 ||
            r.response.at("output")[0].value("type", std::string()) != "reasoning") {
            std::cerr << "Integration: reasoning-only EOS lost its completed reasoning output\n";
            return false;
        }
    }

    // 2. Reasoning followed by a final answer: normal completion
    {
        auto r = run_generation("Finished reasoning.\n\n</think>\n\nDone.\n\n", STOP_TYPE_EOS, false);
        if (!end_turn_true(r.response, r.events)) {
            std::cerr << "Integration: answered EOS set an unexpected end_turn\n";
            return false;
        }
        if (r.msg.content.find("Done.") == std::string::npos || r.msg.reasoning_content.find("Finished reasoning.") == std::string::npos) {
            std::cerr << "Integration: answered EOS parser corrupted reasoning/content\n";
            return false;
        }
    }

    // 3. Reasoning followed by a tool call: normal tool-call lifecycle
    {
        const std::string generated =
            "A tool is needed.\n\n</think>\n\n"
            "<tool_call>\n"
            "{\"name\":\"shell_command\",\"arguments\":{\"command\":\"pwd\"}}\n"
            "</tool_call>";
        auto r = run_generation(generated, STOP_TYPE_EOS, false);
        if (!end_turn_true(r.response, r.events)) {
            std::cerr << "Integration: tool-call EOS set an unexpected end_turn\n";
            return false;
        }
        if (r.msg.tool_calls.size() != 1 || r.msg.tool_calls[0].name != "shell_command") {
            std::cerr << "Integration: tool-call EOS parser did not recover the tool call\n";
            return false;
        }
        bool found_tool_item = false;
        for (const auto & item : r.response.at("output")) {
            if (item.value("type", std::string()) == "function_call" &&
                item.value("name", std::string()) == "shell_command") {
                found_tool_item = true;
            }
        }
        if (!found_tool_item) {
            std::cerr << "Integration: tool-call EOS lost the function_call output item\n";
            return false;
        }
    }

    // 4. Generation stopped by the token limit: no false continuation
    {
        auto r = run_generation("Thinking was interrupted.\n\n", STOP_TYPE_LIMIT, false);
        if (!end_turn_true(r.response, r.events)) {
            std::cerr << "Integration: reasoning-only LIMIT set end_turn=false\n";
            return false;
        }
    }

    // 5. Compaction response: no false continuation
    {
        auto r = run_generation("Summarizing.\n\n", STOP_TYPE_EOS, true);
        if (!end_turn_true(r.response, r.events)) {
            std::cerr << "Integration: compaction EOS set end_turn=false\n";
            return false;
        }
    }

    // 6. Multiple consecutive reasoning-only turns: each continues, reasoning in order
    {
        const std::vector<std::string> turns = { "Turn one reasoning.", "Turn two reasoning.", "Turn three reasoning." };
        std::string seen_reasoning;
        for (const auto & turn : turns) {
            auto r = run_generation(turn + "\n</think>\n\n", STOP_TYPE_EOS, false);
            if (!end_turn_false(r.response, r.events)) {
                std::cerr << "Integration: consecutive reasoning-only turn did not set end_turn=false\n";
                return false;
            }
            if (r.msg.reasoning_content.find(turn) == std::string::npos) {
                std::cerr << "Integration: consecutive reasoning-only turn lost its reasoning\n";
                return false;
            }
            if (!seen_reasoning.empty()) {
                seen_reasoning += "\n\n";
            }
            seen_reasoning += turn;
        }
        // replay the three reasoning items through the bridge, they must stay in order
        json input = json::array();
        for (const auto & turn : turns) {
            input.push_back(json {
                {"type", "reasoning"},
                {"summary", json::array()},
                {"content", json::array({json {
                    {"type", "reasoning_text"},
                    {"text", turn},
                }})},
            });
        }
        const auto converted = server_chat_convert_responses_to_chatcmpl(json {
            {"input", input},
        });
        if (!converted.contains("messages") ||
            converted.at("messages").size() != 1 ||
            converted.at("messages")[0].value("reasoning_content", std::string()) != seen_reasoning) {
            std::cerr << "Integration: consecutive reasoning-only turns were not preserved in order\n";
            return false;
        }
    }

    return true;
}

// The non-incremental final streaming path emits response.output_item.done for a
// tool_search_call without a preceding response.output_item.added. This matches the
// long-standing upstream behavior for function_call (also done-only in this path),
// which Codex consumes without issue. Pin the event sequence so any change is deliberate.
static bool test_responses_tool_search_streaming_lifecycle() {
    server_task_result_cmpl_final result {};
    result.oaicompat_model = "test-model";
    result.oai_resp_id = "resp_tool_search_stream";
    result.generation_params.responses_tool_search = true;
    result.oaicompat_msg.role = "assistant";
    result.oaicompat_msg.tool_calls.push_back({
        "tool_search",
        R"({"query":"calendar"})",
        "search_stream",
    });

    const json events = result.to_json_oaicompat_resp_stream();

    bool found_done = false;
    bool found_added = false;
    std::string done_status;
    for (const auto & event : events) {
        const std::string name = event.value("event", std::string());
        if (name != "response.output_item.done" && name != "response.output_item.added") {
            continue;
        }
        if (!event.contains("data") || !event.at("data").contains("item")) {
            continue;
        }
        const auto & item = event.at("data").at("item");
        if (item.value("type", std::string()) != "tool_search_call") {
            continue;
        }
        if (name == "response.output_item.done") {
            found_done = true;
            done_status = item.value("status", std::string());
        } else {
            found_added = true;
        }
    }

    if (!found_done) {
        std::cerr << "tool_search streaming: missing output_item.done for tool_search_call\n";
        return false;
    }
    if (done_status != "completed") {
        std::cerr << "tool_search streaming: tool_search_call done item not completed\n";
        return false;
    }
    // the done-only sequence (no added) is the established upstream behavior for
    // atomic tool items; assert it stays that way
    if (found_added) {
        std::cerr << "tool_search streaming: unexpected output_item.added for tool_search_call\n";
        return false;
    }

    // the final response must carry the tool_search_call item
    const json & final_response = events.back().at("data").at("response");
    bool in_final = false;
    for (const auto & item : final_response.at("output")) {
        if (item.value("type", std::string()) == "tool_search_call") {
            in_final = true;
        }
    }
    if (!in_final) {
        std::cerr << "tool_search streaming: final response lost the tool_search_call item\n";
        return false;
    }

    return true;
}

static bool test_responses_compaction_update() {
    const std::string template_path = "models/templates/llama-cpp-qwen-based-codex.jinja";
    std::ifstream file(template_path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open template: " << template_path << "\n";
        return false;
    }
    const std::string template_text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto tmpls = common_chat_templates_ptr(common_chat_templates_init(nullptr, template_text));

    common_chat_msg system;
    system.role = "system";
    system.content = "Stable Codex instructions.";
    common_chat_msg user;
    user.role = "user";
    user.content = "Summarize the current task.";
    common_chat_tool shell_tool;
    shell_tool.name = "shell_command";
    shell_tool.description = "Runs a shell command";
    shell_tool.parameters = R"({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]})";

    common_chat_templates_inputs inputs;
    inputs.messages = {system, user};
    inputs.tools = {shell_tool};
    inputs.enable_thinking = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.chat_template_kwargs["is_compaction"] = "true";

    struct generation {
        server_task_result_cmpl_final result;
        json events = json::array();
        std::string prefill;
    };
    auto run = [&](bool stream, const std::vector<std::string> & chunks, const std::string & prefill) {
        auto session_inputs = inputs;
        if (!prefill.empty()) {
            common_chat_msg assistant;
            assistant.role = "assistant";
            assistant.content = prefill;
            session_inputs.messages.push_back(assistant);
            session_inputs.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
        }
        common_chat_session session(tmpls.get(), nullptr, session_inputs);
        const std::string parsed_prefill = session.parse_prefill().content;
        if (session.format() != COMMON_CHAT_FORMAT_PEG_NATIVE ||
            (!prefill.empty() && (parsed_prefill.size() < prefill.size() ||
                                 parsed_prefill.compare(parsed_prefill.size() - prefill.size(), prefill.size(), prefill) != 0))) {
            throw std::runtime_error("compaction test session has an unexpected parser or prefill");
        }
        task_result_state state(std::move(session));
        generation out;
        out.prefill = parsed_prefill;
        std::string generated;
        int n_chunks = 0;
        for (const auto & chunk : chunks) {
            generated += chunk;
            if (stream) {
                server_task_result_cmpl_partial partial {};
                partial.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
                partial.n_decoded = ++n_chunks;
                partial.content = common_chat_input(chunk);
                partial.update(state);
                for (const auto & event : partial.to_json()) {
                    out.events.push_back(event);
                }
            }
        }

        out.result = make_response_result("");
        out.result.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
        out.result.stop = STOP_TYPE_EOS;
        out.result.stream = stream;
        out.result.generation_params.stream = stream;
        out.result.generation_params.responses_compaction = true;
        // Streaming finals carry no text; partial.update() already fed every chunk.
        out.result.content = stream ? common_chat_input() : common_chat_input(generated);
        out.result.update(state);
        return out;
    };

    auto valid = [&](const std::string & name, bool stream, const std::vector<std::string> & chunks,
                     const std::string & summary, const std::string & prefill = "") {
        auto fail = [&](const std::string & detail) {
            std::cerr << "Compaction " << name << ": " << detail << "\n";
            return false;
        };
        try {
            auto out = run(stream, chunks, prefill);
            if (out.result.oai_resp_compaction_content != summary ||
                out.result.oaicompat_msg.content != out.prefill + summary) {
                return fail("final.update() did not extract the exact summary");
            }
            if (prefill.empty() && out.result.oaicompat_msg.reasoning_content.find("Preserve the completed work.") == std::string::npos) {
                return fail("production parser did not separate the reasoning from the summary");
            }
            json response;
            json added;
            json done;
            if (stream) {
                for (const auto & event : out.result.to_json()) {
                    out.events.push_back(event);
                }
                int n_completed = 0;
                int n_added = 0;
                int n_done = 0;
                for (const auto & event : out.events) {
                    const auto type = event.value("event", std::string());
                    const auto & data = event.at("data");
                    if (type == "response.completed") {
                        ++n_completed;
                        response = data.at("response");
                    } else if ((type == "response.output_item.added" || type == "response.output_item.done") &&
                               data.at("item").value("type", std::string()) == "compaction") {
                        if (type == "response.output_item.added") {
                            ++n_added;
                            added = data.at("item");
                        } else {
                            ++n_done;
                            done = data.at("item");
                        }
                    }
                }
                if (n_completed != 1 || n_added != 1 || n_done != 1 || added != done ||
                    out.events.back().at("event") != "response.completed") {
                    return fail("expected one compaction lifecycle followed by response.completed");
                }
                if (format_oai_resp_sse(out.events).find("event: response.completed\n") == std::string::npos) {
                    return fail("SSE output lost response.completed");
                }
            } else {
                response = out.result.to_json();
            }
            if (response.at("status") != "completed" || !response.value("end_turn", true)) {
                return fail("compaction response is not completed");
            }
            int n_compaction = 0;
            for (const auto & item : response.at("output")) {
                if (item.value("type", std::string()) != "compaction") {
                    continue;
                }
                ++n_compaction;
                if (item.at("encrypted_content") != "llama.cpp.compaction.v1\n" + summary ||
                    item.at("id").get<std::string>().find("cmp_") != 0 || (stream && item != done)) {
                    return fail("durable payload differs from the extracted summary or stream item");
                }
            }
            if (n_compaction != 1) {
                return fail("expected exactly one compaction item in the completed response");
            }
            std::cout << "Compaction " << name << " passed\n";
            return true;
        } catch (const std::exception & e) {
            return fail(e.what());
        }
    };
    auto invalid = [&](const std::string & name, bool stream, const std::string & generated, const std::string & prefill = "") {
        try {
            run(stream, {generated}, prefill);
            std::cerr << "Compaction " << name << ": final.update() accepted an invalid summary\n";
            return false;
        } catch (const std::runtime_error & e) {
            if (std::string(e.what()) != "Responses compaction generated no model output") {
                std::cerr << "Compaction " << name << ": unexpected rejection: " << e.what() << "\n";
                return false;
            }
            std::cout << "Compaction " << name << " rejected by final.update()\n";
            return true;
        }
    };

    const std::string reasoning = "Preserve the completed work.\n</think>\n\n";
    const std::string summary = "## Goal\nFix streaming compaction.\n\n- Progress: parser checked.\n- Next: run tests!\n";
    const std::vector<std::string> chunks = {reasoning.substr(0, 30), reasoning.substr(30),
                                           summary.substr(0, 11), summary.substr(11, 27), summary.substr(38)};
    bool ok = valid("streaming empty final chunk", true, chunks, summary);
    ok = valid("non-streaming", false, chunks, summary) && ok;
    std::string long_summary = "## Detailed checkpoint\n";
    while (long_summary.size() < 128*1024) {
        long_summary += "- Preserve paths, commands, decisions and next steps exactly; retain every Markdown line.\n";
    }
    ok = valid("long streaming summary", true, {reasoning, long_summary}, long_summary) && ok;
    ok = valid("long non-streaming summary", false, {reasoning, long_summary}, long_summary) && ok;
    for (const bool stream : {false, true}) {
        const std::string mode = stream ? "streaming " : "non-streaming ";
        ok = invalid(mode + "empty", stream, "") && ok;
        ok = invalid(mode + "whitespace only", stream, " \t\r\n") && ok;
        ok = invalid(mode + "open reasoning only", stream, "Still thinking.") && ok;
        ok = invalid(mode + "closed reasoning only", stream, reasoning) && ok;
        ok = invalid(mode + "whitespace summary", stream, reasoning + " \t\r\n") && ok;

        const std::string prefill = "## Generation prefill\n";
        ok = valid(mode + "prefill removal", stream, {summary.substr(0, 11), summary.substr(11)}, summary, prefill) && ok;
        ok = invalid(mode + "prefill only", stream, "", prefill) && ok;
        ok = invalid(mode + "prefill and whitespace", stream, " \t\r\n", prefill) && ok;

        // These are text, not the tab/CR/newline escape sequences used by validation.
        for (const std::string literal : {"t", "r", "n", "\\", "\\n"}) {
            ok = valid(mode + "literal " + literal, stream, {reasoning, literal}, literal) && ok;
        }
    }
    const std::string generated = reasoning + summary;
    for (const size_t size : {size_t(1), size_t(2), size_t(7), size_t(16), size_t(63)}) {
        std::vector<std::string> split;
        for (size_t pos = 0; pos < generated.size(); pos += size) {
            split.push_back(generated.substr(pos, size));
        }
        ok = valid("chunk size " + std::to_string(size), true, split, summary) && ok;
    }
    return ok;
}

static bool test_responses_stream_output_indices() {
    for (const bool reasoning : {false, true}) {
        for (const bool text : {false, true}) {
            auto result = make_response_result(text ? "Visible answer." : "");
            result.oai_resp_reasoning_id = "rs_indices";
            result.oaicompat_msg.reasoning_content = reasoning ? "Thinking." : "";
            result.generation_params.responses_custom_tools.insert("apply_patch");
            result.generation_params.responses_tool_search = true;
            result.generation_params.responses_compaction = true;
            result.oai_resp_compaction_content = "Saved progress.";
            result.oaicompat_msg.tool_calls = {
                {"exec_command", R"({"cmd":"pwd"})", "exec_indices"},
                {"apply_patch", R"({"input":"*** Begin Patch\n*** End Patch"})", "patch_indices"},
                {"tool_search", R"({"query":"calendar"})", "search_indices"},
                {"exec_command", R"({"cmd":"git status --short"})", "exec_after_search_indices"},
            };
            const auto events = result.to_json_oaicompat_resp_stream();
            const auto & output = events.back().at("data").at("response").at("output");
            auto check_index = [&](const json & event) {
                const auto & data = event.at("data");
                const int index = data.value("output_index", -1);
                if (index < 0 || static_cast<size_t>(index) >= output.size()) {
                    std::cerr << "Responses stream event lost its output index: " << event.dump() << '\n';
                    return false;
                }
                const std::string id = data.contains("item") ? data.at("item").at("id").get<std::string>() : data.value("item_id", std::string());
                if (id != output.at(index).at("id").get<std::string>()) {
                    std::cerr << "Responses stream index points to a different output item\n";
                    return false;
                }
                return true;
            };
            size_t done = 0;
            for (const auto & event : events) {
                const auto & data = event.at("data");
                if (event.at("event") == "response.completed") {
                    continue;
                }
                if (!check_index(event)) {
                    return false;
                }
                if (event.at("event") == "response.output_item.done") {
                    if (data.at("output_index").get<size_t>() != done++) {
                        std::cerr << "Responses stream output indices are not consecutive\n";
                        return false;
                    }
                }
            }
            if (done != output.size()) {
                return false;
            }

            server_task_result_cmpl_partial partial {};
            partial.oai_resp_created = true;
            partial.oai_resp_reasoning_id = result.oai_resp_reasoning_id;
            partial.oai_resp_message_id = result.oai_resp_message_id;
            partial.anthropic_has_reasoning = reasoning;
            partial.responses_custom_tools = result.generation_params.responses_custom_tools;
            partial.responses_tool_search = result.generation_params.responses_tool_search;
            common_chat_msg_diff diff;
            diff.reasoning_content_delta = result.oaicompat_msg.reasoning_content;
            diff.content_delta = result.oaicompat_msg.content;
            diff.tool_call_index = 0;
            diff.tool_call_delta = result.oaicompat_msg.tool_calls.front();
            partial.oaicompat_msg_diffs.push_back(diff);
            for (size_t i = 1; i < result.oaicompat_msg.tool_calls.size(); ++i) {
                common_chat_msg_diff tool_diff;
                tool_diff.tool_call_index = static_cast<int>(i);
                tool_diff.tool_call_delta = result.oaicompat_msg.tool_calls[i];
                partial.oaicompat_msg_diffs.push_back(tool_diff);
            }
            const auto partial_events = partial.to_json_oaicompat_resp();
            for (const auto & event : partial_events) {
                if (!check_index(event)) {
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    if (!test_responses_stream_output_indices()) {
        return 1;
    }

    if (!test_responses_compaction_update()) {
        return 1;
    }

    if (!test_responses_reasoning_replay()) {
        return 1;
    }

    if (!test_responses_reasoning_parser_integration()) {
        return 1;
    }

    if (!test_responses_tool_search_streaming_lifecycle()) {
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
