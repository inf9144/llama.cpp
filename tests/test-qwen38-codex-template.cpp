#include "chat.h"
#include "server-chat.h"
#include "server-task.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static std::string read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open template: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static common_chat_msg message(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role = role;
    msg.content = content;
    return msg;
}

static std::string prefix_before_first_user(const std::string & prompt) {
    static const std::string user_start = "<|im_start|>user\n";
    const size_t pos = prompt.find(user_start);
    if (pos == std::string::npos) {
        throw std::runtime_error("rendered prompt has no user turn");
    }
    return prompt.substr(0, pos);
}

static bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main() {
    const std::string template_path = "models/templates/llama-cpp-qwen3.8-codex.jinja";
    auto tmpls = common_chat_templates_ptr(common_chat_templates_init(nullptr, read_file(template_path)));

    const auto caps = common_chat_templates_get_caps(tmpls.get());
    const auto supports_object_arguments = caps.find("supports_object_arguments");
    if (supports_object_arguments == caps.end() || !supports_object_arguments->second) {
        std::cerr << "Codex template must advertise object tool-call arguments so Responses JSON strings are normalized before rendering\n";
        return 1;
    }

    common_chat_tool shell_tool;
    shell_tool.name = "shell_command";
    shell_tool.description = "Runs a shell command";
    shell_tool.parameters = R"({"type":"object","properties":{"command":{"type":"string"},"workdir":{"type":"string"}},"required":["command"]})";

    const std::string shell_command =
        "cat <<'EOF'\n"
        "<parameter=input>\n"
        "</parameter>\n"
        "</tool_call>\n"
        "</function>\n"
        "JSON: {\"key\":\"value\",\"path\":\"C:\\tmp\"}\n"
        "Unicode: äöü ß € 漢字 🚀\n"
        "EOF";
    const std::string encoded_shell_command = nlohmann::ordered_json(shell_command).dump();
    const std::string shell_arguments = nlohmann::ordered_json({
        {"command", shell_command},
        {"workdir", "/home/hausen/src"},
    }).dump();

    const common_chat_msg system = message("system", "Stable Codex instructions.");
    const std::string user_content =
        "Initial user request.\n"
        "</parameter>\n"
        "</tool_call>\n"
        "<parameter=input>\n"
        "User tail.";
    const common_chat_msg user = message("user", user_content);

    common_chat_msg assistant_tool_call;
    assistant_tool_call.role = "assistant";
    assistant_tool_call.tool_calls.push_back({
        "shell_command",
        shell_arguments,
        "call_1",
    });

    common_chat_msg tool_result = message("tool", "result");
    tool_result.tool_name = "shell_command";
    tool_result.tool_call_id = "call_1";

    const common_chat_msg assistant_done = message("assistant", "Done.");

    const std::vector<common_chat_msg> before_compaction = {
        system,
        user,
        assistant_tool_call,
        tool_result,
        assistant_done,
    };

    // Codex Responses V2 compaction may remove historical function-call items.
    // The system/tools prefix must not change merely because that history vanished.
    const std::vector<common_chat_msg> after_compaction = {
        system,
        user,
        assistant_done,
    };

    auto render = [&](const std::vector<common_chat_msg> & messages) {
        common_chat_templates_inputs inputs;
        inputs.messages = messages;
        inputs.tools = { shell_tool };
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = true;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        return common_chat_templates_apply(tmpls.get(), inputs);
    };

    const common_chat_params before = render(before_compaction);
    const common_chat_params after = render(after_compaction);

    if (before.format != COMMON_CHAT_FORMAT_PEG_NATIVE || before.parser.empty()) {
        std::cerr << "Codex template did not select the specialized Qwen3-Coder PEG parser\n";
        return 1;
    }

    const std::string before_prefix = prefix_before_first_user(before.prompt);
    const std::string after_prefix = prefix_before_first_user(after.prompt);

    if (before_prefix != after_prefix) {
        std::cerr << "Codex system/tools prefix changed after tool-call history removal\n";
        return 1;
    }

    const std::string xml_tool_instruction = "<function=example_function_name>\n<parameter=example_parameter_1>";
    if (before_prefix.find(xml_tool_instruction) == std::string::npos) {
        std::cerr << "Codex template did not default to the Qwen3-Coder XML tool-call format\n";
        return 1;
    }

    const std::string json_tool_instruction =
        "{\"name\": \"example_function_name\", \"arguments\":";
    if (before_prefix.find(json_tool_instruction) != std::string::npos) {
        std::cerr << "Codex template mixed JSON tool-call instructions with the XML Qwen3-Coder parser\n";
        return 1;
    }

    const std::string rendered_user =
        "<|im_start|>user\n" + user_content + "<|im_end|>\n";
    if (before.prompt.find(rendered_user) == std::string::npos) {
        std::cerr << "Literal XML delimiter text in a user message was not preserved byte-for-byte\n";
        return 1;
    }

    const std::string rendered_command =
        "<parameter=command>\n" + encoded_shell_command + "\n</parameter>";
    if (before.prompt.find(rendered_command) == std::string::npos) {
        std::cerr << "Historical string arguments were not JSON-escaped inside XML parameters\n";
        return 1;
    }

    // Exercise the same specialized PEG parser used by llama-server. The model
    // generates after the already-prefilled '<think>\n', so this string starts
    // with reasoning content and then closes the thinking block before the call.
    common_chat_parser_params parser_params(before);
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    parser_params.parser.load(before.parser);

    const std::string generated =
        "Inspecting the requested file.\n</think>\n\n"
        "<tool_call>\n"
        "<function=shell_command>\n"
        "<parameter=command>\n" + encoded_shell_command + "\n</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const common_chat_msg parsed = common_chat_parse(generated, false, parser_params);
    if (parsed.tool_calls.size() != 1 || parsed.tool_calls[0].name != "shell_command") {
        std::cerr << "Qwen3-Coder PEG parser did not recover the shell_command tool call\n";
        return 1;
    }

    nlohmann::ordered_json parsed_arguments;
    try {
        parsed_arguments = nlohmann::ordered_json::parse(parsed.tool_calls[0].arguments);
    } catch (const std::exception & e) {
        std::cerr << "Qwen3-Coder PEG parser produced invalid JSON arguments: " << e.what() << "\n";
        return 1;
    }

    if (!parsed_arguments.contains("command") ||
        !parsed_arguments.at("command").is_string() ||
        parsed_arguments.at("command").get<std::string>() != shell_command) {
        std::cerr << "Qwen3-Coder PEG parser corrupted quoted shell command arguments\n";
        return 1;
    }

    // Responses custom/freeform tools wrap the raw payload in input.data. The
    // outer object forces Qwen3-Coder onto its JSON-value PEG path, while the
    // nested string schema still guarantees that the transported value is text.
    common_chat_tool custom_tool;
    custom_tool.name = "apply_patch";
    custom_tool.description = "Responses custom/freeform transport test";
    custom_tool.parameters = R"({"type":"object","properties":{"input":{"type":"object","properties":{"data":{"type":"string"}},"required":["data"],"additionalProperties":false}},"required":["input"],"additionalProperties":false})";

    const std::string custom_input =
        "*** Begin Patch\n"
        "*** Add File: xml_delim_test.txt\n"
        "+<function=apply_patch>\n"
        "+<parameter=input>\n"
        "+</parameter>\n"
        "+</function>\n"
        "+quote: \"hello\" backslash: C:\\tmp\\x unicode: Ä € 🚀\n"
        "*** End Patch";
    const std::string encoded_custom_input = nlohmann::ordered_json({{"data", custom_input}}).dump();

    common_chat_templates_inputs custom_inputs;
    custom_inputs.messages = { system, user };
    custom_inputs.tools = { custom_tool };
    custom_inputs.add_generation_prompt = true;
    custom_inputs.enable_thinking = true;
    custom_inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const common_chat_params custom_params = common_chat_templates_apply(tmpls.get(), custom_inputs);

    if (custom_params.format != COMMON_CHAT_FORMAT_PEG_NATIVE || custom_params.parser.empty()) {
        std::cerr << "Custom/freeform test did not select the specialized Qwen3-Coder PEG parser\n";
        return 1;
    }

    common_chat_parser_params custom_parser_params(custom_params);
    custom_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    custom_parser_params.parser.load(custom_params.parser);

    const std::string generated_custom =
        "Preparing a literal delimiter patch.\n</think>\n\n"
        "<tool_call>\n"
        "<function=apply_patch>\n"
        "<parameter=input>\n" + encoded_custom_input + "\n</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const common_chat_msg parsed_custom = common_chat_parse(generated_custom, false, custom_parser_params);
    if (parsed_custom.tool_calls.size() != 1 || parsed_custom.tool_calls[0].name != "apply_patch") {
        std::cerr << "Qwen3-Coder PEG parser did not recover the custom/freeform tool call\n";
        return 1;
    }

    nlohmann::ordered_json parsed_custom_arguments;
    try {
        parsed_custom_arguments = nlohmann::ordered_json::parse(parsed_custom.tool_calls[0].arguments);
    } catch (const std::exception & e) {
        std::cerr << "Custom/freeform PEG transport produced invalid JSON arguments: " << e.what() << "\n";
        return 1;
    }

    if (!parsed_custom_arguments.contains("input") ||
        !parsed_custom_arguments.at("input").is_object() ||
        !parsed_custom_arguments.at("input").contains("data") ||
        !parsed_custom_arguments.at("input").at("data").is_string() ||
        parsed_custom_arguments.at("input").at("data").get<std::string>() != custom_input) {
        std::cerr << "Custom/freeform PEG transport corrupted literal XML delimiter content\n";
        return 1;
    }

    // Match the Responses ingress representation for historical custom_tool_call.
    // Parse the rendered parameter semantically instead of assuming a particular
    // tojson whitespace layout. A raw framing leak would introduce an actual
    // newline before </parameter> and therefore truncate this JSON value.
    common_chat_msg historical_custom_call;
    historical_custom_call.role = "assistant";
    historical_custom_call.tool_calls.push_back({
        "apply_patch",
        nlohmann::ordered_json({{"input", {{"data", custom_input}}}}).dump(),
        "call_custom",
    });
    common_chat_msg historical_custom_result = message("tool", "Done!");
    historical_custom_result.tool_name = "apply_patch";
    historical_custom_result.tool_call_id = "call_custom";

    custom_inputs.messages = {
        system,
        user,
        historical_custom_call,
        historical_custom_result,
        assistant_done,
    };
    const common_chat_params historical_custom = common_chat_templates_apply(tmpls.get(), custom_inputs);
    const std::string historical_marker = "<function=apply_patch>\n<parameter=input>\n";
    const size_t historical_begin = historical_custom.prompt.find(historical_marker);
    if (historical_begin == std::string::npos) {
        std::cerr << "Historical custom/freeform replay did not render an input parameter\n";
        return 1;
    }
    const size_t historical_value_begin = historical_begin + historical_marker.size();
    const size_t historical_end = historical_custom.prompt.find("\n</parameter>", historical_value_begin);
    if (historical_end == std::string::npos) {
        std::cerr << "Historical custom/freeform replay did not close the input parameter\n";
        return 1;
    }

    nlohmann::ordered_json historical_value;
    try {
        historical_value = nlohmann::ordered_json::parse(
            historical_custom.prompt.substr(historical_value_begin, historical_end - historical_value_begin));
    } catch (const std::exception & e) {
        std::cerr << "Historical custom/freeform input leaked through XML framing: " << e.what() << "\n";
        return 1;
    }

    if (!historical_value.is_object() || historical_value.size() != 1 ||
        !historical_value.contains("data") || !historical_value.at("data").is_string() ||
        historical_value.at("data").get<std::string>() != custom_input) {
        std::cerr << "Historical custom/freeform replay corrupted the transported payload\n";
        return 1;
    }

    // Codex 0.147 client-side tool_search is bridged through an internal
    // function, while discovered tools remain out of the stable top-level tool
    // block and are carried separately for parser/grammar expansion.
    const nlohmann::ordered_json tool_search_request = {
        {"model", "test-model"},
        {"input", "Find calendar tools"},
        {"tools", nlohmann::ordered_json::array({
            {
                {"type", "tool_search"},
                {"execution", "client"},
                {"description", "Search deferred tools"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"query", {{"type", "string"}}}}},
                    {"required", nlohmann::ordered_json::array({"query"})},
                    {"additionalProperties", false},
                }},
            },
            {{"type", "web_search"}},
        })},
    };
    const auto converted_search = server_chat_convert_responses_to_chatcmpl(tool_search_request);
    if (!converted_search.value("__llamacpp_responses_tool_search", false) ||
        !converted_search.contains("tools") || converted_search.at("tools").size() != 1 ||
        converted_search.at("tools")[0]["function"]["name"] != "tool_search") {
        std::cerr << "Responses tool_search was not exposed as the sole internal client tool\n";
        return 1;
    }

    const nlohmann::ordered_json followup_request = {
        {"model", "test-model"},
        {"input", nlohmann::ordered_json::array({
            {
                {"type", "message"},
                {"role", "user"},
                {"content", nlohmann::ordered_json::array({{{"type", "input_text"}, {"text", "Find calendar tools"}}})},
            },
            {
                {"type", "tool_search_call"},
                {"call_id", "search-1"},
                {"execution", "client"},
                {"arguments", {{"query", "calendar"}}},
            },
            {
                {"type", "tool_search_output"},
                {"call_id", "search-1"},
                {"status", "completed"},
                {"execution", "client"},
                {"tools", nlohmann::ordered_json::array({
                    {
                        {"type", "namespace"},
                        {"name", "mcp__calendar"},
                        {"description", "Calendar tools"},
                        {"tools", nlohmann::ordered_json::array({
                            {
                                {"type", "function"},
                                {"name", "create_event"},
                                {"description", "Create an event"},
                                {"defer_loading", true},
                                {"parameters", {
                                    {"type", "object"},
                                    {"properties", {{"title", {{"type", "string"}}}}},
                                    {"required", nlohmann::ordered_json::array({"title"})},
                                    {"additionalProperties", false},
                                }},
                            },
                        })},
                    },
                })},
            },
        })},
        {"tools", tool_search_request.at("tools")},
    };
    const auto converted_followup = server_chat_convert_responses_to_chatcmpl(followup_request);
    if (!converted_followup.contains("__llamacpp_responses_deferred_tools") ||
        converted_followup.at("__llamacpp_responses_deferred_tools").size() != 1 ||
        converted_followup.at("__llamacpp_responses_deferred_tools")[0]["function"]["name"] !=
            "mcp__calendar.create_event") {
        std::cerr << "tool_search_output did not expose the discovered namespaced tool to the parser layer\n";
        return 1;
    }
    for (const auto & tool : converted_followup.at("tools")) {
        if (tool["function"]["name"] == "mcp__calendar.create_event") {
            std::cerr << "Deferred tool leaked into the stable top-level tools block\n";
            return 1;
        }
    }

    const auto & followup_messages = converted_followup.at("messages");
    if (followup_messages.size() < 3 ||
        followup_messages[1]["tool_calls"][0]["function"]["name"] != "tool_search" ||
        followup_messages[2]["role"] != "tool" ||
        followup_messages[2]["content"].get<std::string>().find("mcp__calendar.create_event") == std::string::npos) {
        std::cerr << "tool_search call/output history was not preserved in chronological model context\n";
        return 1;
    }

    common_chat_tool tool_search_tool;
    tool_search_tool.name = "tool_search";
    tool_search_tool.description = "Search deferred tools";
    tool_search_tool.parameters = R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false})";

    common_chat_tool deferred_tool;
    deferred_tool.name = "mcp__calendar.create_event";
    deferred_tool.description = "Create an event";
    deferred_tool.parameters = R"({"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false})";

    common_chat_msg search_call;
    search_call.role = "assistant";
    search_call.tool_calls.push_back({
        "tool_search",
        nlohmann::ordered_json({{"query", "calendar"}}).dump(),
        "search-1",
    });
    common_chat_msg search_result = message(
        "tool",
        nlohmann::ordered_json({{"tools", converted_followup.at("__llamacpp_responses_deferred_tools")}}).dump());
    search_result.tool_call_id = "search-1";

    common_chat_templates_inputs stable_inputs;
    stable_inputs.messages = { system, user, search_call, search_result };
    stable_inputs.tools = { tool_search_tool };
    stable_inputs.add_generation_prompt = true;
    stable_inputs.enable_thinking = true;
    stable_inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const auto stable_params = common_chat_templates_apply(tmpls.get(), stable_inputs);

    auto parser_inputs = stable_inputs;
    parser_inputs.tools.push_back(deferred_tool);
    const auto expanded_parser_params = common_chat_templates_apply(tmpls.get(), parser_inputs);
    if (stable_params.generation_prompt != expanded_parser_params.generation_prompt) {
        std::cerr << "Deferred tool expansion changed the Qwen generation prefix\n";
        return 1;
    }
    const std::string stable_prefix = prefix_before_first_user(stable_params.prompt);
    if (stable_prefix.find("mcp__calendar.create_event") != std::string::npos) {
        std::cerr << "Deferred tool polluted the cache-stable Qwen system/tools prefix\n";
        return 1;
    }
    if (stable_params.prompt.find("mcp__calendar.create_event") == std::string::npos) {
        std::cerr << "Deferred tool was not visible in the chronological tool_search result\n";
        return 1;
    }

    common_chat_parser_params deferred_parser(expanded_parser_params);
    deferred_parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    deferred_parser.parser.load(expanded_parser_params.parser);
    const std::string generated_deferred =
        "Using the discovered calendar tool.\n</think>\n\n"
        "<tool_call>\n"
        "<function=mcp__calendar.create_event>\n"
        "<parameter=title>\n\"Lunch\"\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    const auto parsed_deferred = common_chat_parse(generated_deferred, false, deferred_parser);
    if (parsed_deferred.tool_calls.size() != 1 ||
        parsed_deferred.tool_calls[0].name != "mcp__calendar.create_event") {
        std::cerr << "Expanded parser did not accept the deferred MCP tool\n";
        return 1;
    }

    auto repeated_search_request = followup_request;
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

    // Verify that an internally parsed tool_search function call is emitted
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

    // Qwen3.8 reasoning-effort compatibility. Medium is the model-native
    // unsteered baseline; high is the bounded anti-overthinking mode used by
    // default for this Codex-oriented template.
    const std::string low_reasoning_instruction =
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    const std::string high_reasoning_instruction =
        "Reasoning effort is set to high. Think through the task carefully and verify the key points needed for a correct answer. Stay focused, avoid exploring low-value alternatives or repeating settled points, and conclude once the important uncertainties are resolved.";
    const std::string xhigh_reasoning_instruction =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

    auto render_reasoning = [&](const std::vector<common_chat_msg> & messages,
                                const std::string & effort,
                                bool enable_thinking,
                                bool preserve_reasoning) {
        common_chat_templates_inputs inputs;
        inputs.messages = messages;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = enable_thinking;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.chat_template_kwargs["preserve_reasoning"] = preserve_reasoning ? "true" : "false";
        if (!effort.empty()) {
            inputs.chat_template_kwargs["reasoning_effort"] = nlohmann::ordered_json(effort).dump();
        }
        return common_chat_templates_apply(tmpls.get(), inputs);
    };

    const std::vector<common_chat_msg> simple_reasoning_messages = { system, user };
    const auto default_reasoning = render_reasoning(simple_reasoning_messages, "", true, true);
    if (default_reasoning.prompt.find(high_reasoning_instruction) == std::string::npos ||
        default_reasoning.prompt.find(low_reasoning_instruction) != std::string::npos ||
        default_reasoning.prompt.find(xhigh_reasoning_instruction) != std::string::npos) {
        std::cerr << "Qwen3.8 default reasoning effort was not the focused high mode\n";
        return 1;
    }
    if (!ends_with(default_reasoning.prompt, "<|im_start|>assistant\n<think>\n")) {
        std::cerr << "Thinking-enabled Qwen3.8 generation prompt did not leave <think> open\n";
        return 1;
    }

    const auto medium_reasoning = render_reasoning(simple_reasoning_messages, "medium", true, true);
    if (medium_reasoning.prompt.find("Reasoning effort is set to ") != std::string::npos) {
        std::cerr << "Qwen3.8 medium reasoning effort unexpectedly injected steering\n";
        return 1;
    }

    const auto explicit_high = render_reasoning(simple_reasoning_messages, "high", true, true);
    if (explicit_high.prompt.find(high_reasoning_instruction) == std::string::npos) {
        std::cerr << "Qwen3.8 high reasoning effort lost bounded reasoning steering\n";
        return 1;
    }

    for (const std::string effort : {"low", "minimal"}) {
        const auto params = render_reasoning(simple_reasoning_messages, effort, true, true);
        if (params.prompt.find(low_reasoning_instruction) == std::string::npos) {
            std::cerr << "Qwen3.8 low/minimal reasoning alias did not use low steering\n";
            return 1;
        }
    }
    for (const std::string effort : {"xhigh", "max", "ultra"}) {
        const auto params = render_reasoning(simple_reasoning_messages, effort, true, true);
        if (params.prompt.find(xhigh_reasoning_instruction) == std::string::npos) {
            std::cerr << "Qwen3.8 xhigh/max/ultra reasoning alias did not use xhigh steering\n";
            return 1;
        }
    }
    for (const std::string effort : {"none", "off"}) {
        const auto params = render_reasoning(simple_reasoning_messages, effort, true, true);
        if (!ends_with(params.prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n") ||
            params.prompt.find("Reasoning effort is set to ") != std::string::npos) {
            std::cerr << "Qwen3.8 none/off reasoning alias did not disable thinking canonically\n";
            return 1;
        }
    }
    const auto disabled_reasoning = render_reasoning(simple_reasoning_messages, "high", false, true);
    if (!ends_with(disabled_reasoning.prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n") ||
        disabled_reasoning.prompt.find("Reasoning effort is set to ") != std::string::npos) {
        std::cerr << "Explicitly disabled Qwen3.8 thinking did not use the canonical closed block\n";
        return 1;
    }

    bool rejected_unknown_effort = false;
    try {
        (void) render_reasoning(simple_reasoning_messages, "turbo", true, true);
    } catch (const std::exception &) {
        rejected_unknown_effort = true;
    }
    if (!rejected_unknown_effort) {
        std::cerr << "Unknown Qwen3.8 reasoning effort was not rejected\n";
        return 1;
    }

    // Preserve Qwen3.8's canonical history representation. An empty historical
    // thinking block is meaningful for a genuine non-thinking/zero-reasoning
    // turn; stripping old reasoning must omit the whole block instead of
    // replacing non-empty reasoning with a synthetic empty block.
    common_chat_msg historical_reasoned = message("assistant", "Prior answer.");
    historical_reasoned.reasoning_content = "Prior reasoning.";
    common_chat_msg historical_empty = message("assistant", "No hidden reasoning.");
    const common_chat_msg later_user = message("user", "Next question.");

    const auto preserved_reasoned = render_reasoning(
        { system, user, historical_reasoned, later_user }, "medium", true, true);
    if (preserved_reasoned.prompt.find(
            "<|im_start|>assistant\n<think>\nPrior reasoning.\n</think>\n\nPrior answer.<|im_end|>\n") == std::string::npos) {
        std::cerr << "Non-empty historical Qwen3.8 reasoning was not preserved\n";
        return 1;
    }

    const auto preserved_empty = render_reasoning(
        { system, user, historical_empty, later_user }, "medium", true, true);
    if (preserved_empty.prompt.find(
            "<|im_start|>assistant\n<think>\n\n</think>\n\nNo hidden reasoning.<|im_end|>\n") == std::string::npos) {
        std::cerr << "Canonical empty historical Qwen3.8 thinking block was not preserved\n";
        return 1;
    }

    const auto stripped_history = render_reasoning(
        { system, user, historical_reasoned, later_user }, "medium", true, false);
    if (stripped_history.prompt.find("Prior reasoning.") != std::string::npos ||
        stripped_history.prompt.find(
            "<|im_start|>assistant\n<think>\n\n</think>\n\nPrior answer.") != std::string::npos ||
        stripped_history.prompt.find(
            "<|im_start|>assistant\nPrior answer.<|im_end|>\n") == std::string::npos) {
        std::cerr << "preserve_reasoning=false synthesized or retained old thinking history\n";
        return 1;
    }

    common_chat_msg latest_tool_reasoning;
    latest_tool_reasoning.role = "assistant";
    latest_tool_reasoning.reasoning_content = "Latest tool reasoning.";
    latest_tool_reasoning.tool_calls.push_back({
        "shell_command",
        nlohmann::ordered_json({{"command", "pwd"}}).dump(),
        "latest_tool",
    });
    const auto latest_preserved = render_reasoning(
        { system, user, latest_tool_reasoning }, "medium", true, false);
    if (latest_preserved.prompt.find(
            "<|im_start|>assistant\n<think>\nLatest tool reasoning.\n</think>\n\n") == std::string::npos) {
        std::cerr << "Latest agent-loop reasoning was stripped despite preserve_reasoning=false\n";
        return 1;
    }

    std::cout << "Qwen3.8 Codex tool framing, reasoning compatibility, and compaction tests passed\n";
    return 0;
}
