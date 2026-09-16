#include "chat.h"
#include "json.h"
#include "json-schema-to-grammar.h"
#include "server-chat.h"
#include "server-task.h"
#include "../src/llama-grammar.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;

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

static bool grammar_accepts(const std::string & grammar_str, const std::string & input) {
    llama_grammar * grammar = llama_grammar_init_impl(
        nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);
    if (grammar == nullptr) {
        return false;
    }

    bool accepted = true;
    try {
        for (char c : input) {
            // The framing regression corpus below is ASCII after JSON encoding, so
            // feeding one byte at a time also lets us stop immediately on rejection.
            llama_grammar_accept_str(*grammar, std::string(1, c));
            if (llama_grammar_get_stacks(grammar).empty()) {
                accepted = false;
                break;
            }
        }
    } catch (const std::runtime_error &) {
        // llama_grammar_accept_str() throws when a piece makes the grammar stack
        // empty. For negative regression cases that is an expected rejection.
        accepted = false;
    }

    if (accepted) {
        accepted = false;
        for (const auto & stack : llama_grammar_get_stacks(grammar)) {
            if (stack.empty()) {
                accepted = true;
                break;
            }
        }
    }

    llama_grammar_free_impl(grammar);
    return accepted;
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
    const std::string encoded_shell_command = json(shell_command).dump();
    const std::string shell_arguments = json({
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

    json parsed_arguments;
    try {
        parsed_arguments = json::parse(parsed.tool_calls[0].arguments);
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

    // Responses custom/freeform tools use one direct string argument. The
    // llama.cpp:xml-string-args=json template marker forces that string through
    // the JSON-value PEG path, so embedded XML delimiter text remains data.
    common_chat_tool custom_tool;
    custom_tool.name = "apply_patch";
    custom_tool.description = "Responses custom/freeform transport test";
    custom_tool.parameters = R"({"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false})";

    const std::string custom_input =
        "*** Begin Patch\n"
        "*** Add File: xml_delim_test.txt\n"
        "+<function=apply_patch>\n"
        "+<parameter=input>\n"
        "+</parameter>\n"
        "+</function>\n"
        "+quote: \"hello\" backslash: C:\\tmp\\x unicode: Ä € 🚀\n"
        "*** End Patch";
    const std::string encoded_custom_input = json(custom_input).dump();

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

    json parsed_custom_arguments;
    try {
        parsed_custom_arguments = json::parse(parsed_custom.tool_calls[0].arguments);
    } catch (const std::exception & e) {
        std::cerr << "Custom/freeform PEG transport produced invalid JSON arguments: " << e.what() << "\n";
        return 1;
    }

    if (!parsed_custom_arguments.contains("input") ||
        !parsed_custom_arguments.at("input").is_string() ||
        parsed_custom_arguments.at("input").get<std::string>() != custom_input) {
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
        json({{"input", custom_input}}).dump(),
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

    json historical_value;
    try {
        historical_value = json::parse(
            historical_custom.prompt.substr(historical_value_begin, historical_end - historical_value_begin));
    } catch (const std::exception & e) {
        std::cerr << "Historical custom/freeform input leaked through XML framing: " << e.what() << "\n";
        return 1;
    }

    if (!historical_value.is_string() || historical_value.get<std::string>() != custom_input) {
        std::cerr << "Historical custom/freeform replay corrupted the transported payload\n";
        return 1;
    }

    // Codex 0.147 client-side tool_search is bridged through an internal
    // function, while discovered tools remain out of the stable top-level tool
    // block and are carried separately for parser/grammar expansion.
    const json custom_tool_request = {
        {"model", "test-model"},
        {"input", "Apply a patch"},
        {"tools", json::array({
            {
                {"type", "custom"},
                {"name", "apply_patch"},
                {"description", "Apply a patch"},
            },
        })},
    };
    const auto converted_custom_tool = server_chat_convert_responses_to_chatcmpl(custom_tool_request);
    if (!converted_custom_tool.contains("tools") || converted_custom_tool.at("tools").size() != 1 ||
        converted_custom_tool.at("tools")[0]["function"]["parameters"]["properties"]["input"].value("type", std::string()) != "string") {
        std::cerr << "Responses custom/freeform bridge did not expose direct string input\n";
        return 1;
    }

    const json & apply_patch_input_schema =
        converted_custom_tool.at("tools")[0]["function"]["parameters"]["properties"]["input"];
    if (!apply_patch_input_schema.contains("pattern") ||
        !apply_patch_input_schema.at("pattern").is_string()) {
        std::cerr << "Responses apply_patch bridge did not constrain freeform patch framing\n";
        return 1;
    }

    common_chat_tool guarded_custom_tool;
    guarded_custom_tool.name = "apply_patch";
    guarded_custom_tool.description = "Guarded Responses custom/freeform tool";
    guarded_custom_tool.parameters =
        converted_custom_tool.at("tools")[0]["function"]["parameters"].dump();

    common_chat_templates_inputs guarded_inputs;
    guarded_inputs.messages = { system, user };
    guarded_inputs.tools = { guarded_custom_tool };
    guarded_inputs.add_generation_prompt = true;
    guarded_inputs.enable_thinking = true;
    guarded_inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    const common_chat_params guarded_params = common_chat_templates_apply(tmpls.get(), guarded_inputs);

    if (guarded_params.grammar.empty() ||
        guarded_params.grammar.find("*** Begin Patch") == std::string::npos ||
        guarded_params.grammar.find("*** End Patch") == std::string::npos) {
        std::cerr << "Responses apply_patch framing constraint did not reach the Qwen sampling grammar\n";
        return 1;
    }

    // common_chat_parse() intentionally validates the structural PEG only. The
    // JSON-schema pattern is compiled into GBNF by parser.build_grammar() and is
    // enforced during sampling, so exercise that layer directly as well.
    const std::string apply_patch_input_grammar =
        json_schema_to_grammar(apply_patch_input_schema, true);
    if (apply_patch_input_grammar.find("*** Begin Patch") == std::string::npos ||
        apply_patch_input_grammar.find("*** End Patch") == std::string::npos) {
        std::cerr << "Responses apply_patch input pattern did not compile into GBNF framing rules\n";
        return 1;
    }

    common_chat_parser_params guarded_parser_params(guarded_params);
    guarded_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    guarded_parser_params.parser.load(guarded_params.parser);

    auto guarded_parser_accepts = [&](const std::string & input) {
        const std::string generated_guarded =
            "Testing guarded patch framing.\n</think>\n\n"
            "<tool_call>\n"
            "<function=apply_patch>\n"
            "<parameter=input>\n" + json(input).dump() + "\n</parameter>\n"
            "</function>\n"
            "</tool_call>";

        try {
            const common_chat_msg parsed_guarded =
                common_chat_parse(generated_guarded, false, guarded_parser_params);
            return parsed_guarded.tool_calls.size() == 1 &&
                   parsed_guarded.tool_calls[0].name == "apply_patch";
        } catch (const std::exception &) {
            return false;
        }
    };

    auto guarded_grammar_accepts = [&](const std::string & input) {
        return grammar_accepts(apply_patch_input_grammar, json(input).dump());
    };

    const std::vector<std::string> valid_patch_inputs = {
        "*** Begin Patch\n"
        "*** Add File: a.txt\n"
        "+hello\n"
        "*** End Patch",
        "*** Begin Patch\n"
        "*** Delete File: old.txt\n"
        "*** End Patch",
        "*** Begin Patch\n"
        "*** Update File: a.txt\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch",
        "*** Begin Patch\n"
        "*** Add File: escaped.txt\n"
        "+quote: \"hello\" backslash: C:\\tmp\\x\n"
        "*** End Patch",
        "*** Begin Patch\n"
        "*** Update File: literal-marker.txt\n"
        "+*** End Patch\n"
        "*** End Patch",
        "*** Begin Patch\n"
        "*** Update File: eof.txt\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End of File\n"
        "*** End Patch",
        "*** Begin Patch\n"
        "*** Environment ID: test-environment\n"
        "*** Add File: literal-unicode-escape.txt\n"
        "+literal \\u00e9\n"
        "*** End Patch\n",
    };
    for (const std::string & input : valid_patch_inputs) {
        const bool peg_accepts = guarded_parser_accepts(input);
        const bool gbnf_accepts = guarded_grammar_accepts(input);
        if (!peg_accepts || !gbnf_accepts) {
            std::cerr << "Responses apply_patch guard rejected a valid framed payload: "
                      << json(input).dump()
                      << " (PEG=" << (peg_accepts ? "accept" : "reject")
                      << ", GBNF=" << (gbnf_accepts ? "accept" : "reject")
                      << ")\n";
            return 1;
        }
    }

    const std::vector<std::string> invalid_patch_inputs = {
        "",
        ">>> Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch",
        "*** Begin Patch",
        "*** Begin Patch\n*** End Patch",
        "*** Begin Patch\n*** Add File: a.txt\n+hello",
        "*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch\ntrailing",
        "*** Begin Patch\n*** Create File: a.txt\n+hello\n*** End Patch",
        "*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch\n*** Update File: b.txt\n-old\n+new\n*** End Patch",
        "*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch\n<tool_call>\n*** End Patch",
    };
    for (const std::string & input : invalid_patch_inputs) {
        if (guarded_grammar_accepts(input)) {
            std::cerr << "Responses apply_patch sampling grammar accepted invalid framing: "
                      << json(input).dump() << "\n";
            return 1;
        }
    }

    // Exercise the lexical JSON-string layer directly. A Unicode escape that
    // decodes to LF must not create a hidden patch-line boundary after grammar
    // validation.
    const std::string smuggled_newline_json =
        R"("*** Begin Patch\n*** Add File: a.txt\n+hello\u000a*** End Patch\n*** End Patch")";
    if (grammar_accepts(apply_patch_input_grammar, smuggled_newline_json)) {
        std::cerr << "Responses apply_patch sampling grammar accepted an encoded newline smuggle\n";
        return 1;
    }

    json generic_custom_tool_request = custom_tool_request;
    generic_custom_tool_request["tools"][0]["name"] = "freeform_echo";
    const auto converted_generic_custom_tool =
        server_chat_convert_responses_to_chatcmpl(generic_custom_tool_request);
    const json & generic_input_schema =
        converted_generic_custom_tool.at("tools")[0]["function"]["parameters"]["properties"]["input"];
    if (generic_input_schema.contains("pattern")) {
        std::cerr << "Responses bridge unexpectedly constrained a non-apply_patch custom tool\n";
        return 1;
    }

    // The Codex template can explicitly switch to JSON tool calls. The Qwen
    // specialized parser must follow that wire format instead of continuing to
    // expect <function>/<parameter> XML. This is especially important for shell
    // commands containing strings that look like XML closing delimiters.
    common_chat_tool exec_tool;
    exec_tool.name = "exec_command";
    exec_tool.description = "Run a shell command";
    exec_tool.parameters =
        R"({"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"],"additionalProperties":false})";

    common_chat_templates_inputs json_wire_inputs;
    json_wire_inputs.messages = { system, user };
    json_wire_inputs.tools = { exec_tool, guarded_custom_tool };
    json_wire_inputs.add_generation_prompt = true;
    json_wire_inputs.enable_thinking = true;
    json_wire_inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    json_wire_inputs.parallel_tool_calls = true;
    json_wire_inputs.chat_template_kwargs["tool_call_format"] = R"("json")";

    const common_chat_params json_wire_params =
        common_chat_templates_apply(tmpls.get(), json_wire_inputs);

    if (json_wire_params.prompt.find(json_tool_instruction) == std::string::npos ||
        json_wire_params.prompt.find(xml_tool_instruction) != std::string::npos) {
        std::cerr << "Explicit JSON tool-call mode did not render matching Qwen instructions\n";
        return 1;
    }
    if (json_wire_params.grammar.empty() ||
        json_wire_params.grammar.find("*** Begin Patch") == std::string::npos ||
        json_wire_params.grammar.find("*** End Patch") == std::string::npos) {
        std::cerr << "apply_patch framing constraint did not survive JSON tool-call mode\n";
        return 1;
    }

    const std::string valid_json_patch_call =
        "<tool_call>\n"
        "{\"name\":\"apply_patch\",\"arguments\":{\"input\":" +
        json(valid_patch_inputs.front()).dump() +
        "}}\n</tool_call>";
    if (!grammar_accepts(json_wire_params.grammar, valid_json_patch_call)) {
        std::cerr << "Qwen JSON sampling grammar rejected a valid apply_patch call\n";
        return 1;
    }

    const std::string runaway_patch_input =
        "*** Begin Patch\n"
        "*** Add File: a.txt\n"
        "+hello\n"
        "*** End Patch\n"
        "<tool_call>\n"
        "*** End Patch";
    const std::string runaway_json_patch_call =
        "<tool_call>\n"
        "{\"name\":\"apply_patch\",\"arguments\":{\"input\":" +
        json(runaway_patch_input).dump() +
        "}}\n</tool_call>";
    if (grammar_accepts(json_wire_params.grammar, runaway_json_patch_call)) {
        std::cerr << "Qwen JSON sampling grammar accepted content after the first *** End Patch\n";
        return 1;
    }

    common_chat_parser_params json_wire_parser_params(json_wire_params);
    json_wire_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    json_wire_parser_params.parser.load(json_wire_params.parser);

    const std::string json_exec_command =
        "printf '%s\\n' '</parameter>' '</tool_call>'";
    const std::string json_patch_input = valid_patch_inputs.front();
    const std::string generated_json_tools =
        "Using JSON tool calls.\n</think>\n\n"
        "<tool_call>\n"
        "{\"name\":\"exec_command\",\"arguments\":{\"cmd\":" +
        json(json_exec_command).dump() +
        "}}\n</tool_call>\n"
        "<tool_call>\n"
        "{\"name\":\"apply_patch\",\"arguments\":{\"input\":" +
        json(json_patch_input).dump() +
        "}}\n</tool_call>";

    const common_chat_msg parsed_json_tools =
        common_chat_parse(generated_json_tools, false, json_wire_parser_params);
    if (parsed_json_tools.tool_calls.size() != 2 ||
        parsed_json_tools.tool_calls[0].name != "exec_command" ||
        parsed_json_tools.tool_calls[1].name != "apply_patch") {
        std::cerr << "Qwen JSON parser did not recover repeated tool-call wrappers\n";
        return 1;
    }

    json parsed_exec_arguments;
    json parsed_patch_arguments;
    try {
        parsed_exec_arguments = json::parse(parsed_json_tools.tool_calls[0].arguments);
        parsed_patch_arguments = json::parse(parsed_json_tools.tool_calls[1].arguments);
    } catch (const std::exception & e) {
        std::cerr << "Qwen JSON parser produced invalid tool arguments: " << e.what() << "\n";
        return 1;
    }

    if (!parsed_exec_arguments.contains("cmd") ||
        !parsed_exec_arguments.at("cmd").is_string() ||
        parsed_exec_arguments.at("cmd").get<std::string>() != json_exec_command) {
        std::cerr << "Qwen JSON parser corrupted exec_command delimiter text\n";
        return 1;
    }
    if (!parsed_patch_arguments.contains("input") ||
        !parsed_patch_arguments.at("input").is_string() ||
        parsed_patch_arguments.at("input").get<std::string>() != json_patch_input) {
        std::cerr << "Qwen JSON parser corrupted apply_patch input\n";
        return 1;
    }

    const std::string custom_tool_description =
        converted_custom_tool.at("tools")[0]["function"].value("description", std::string());
    if (custom_tool_description.find("model-facing tool-call encoding") == std::string::npos ||
        custom_tool_description.find("model-facing XML transport") != std::string::npos ||
        custom_tool_description.find("Use `*** Add File: <path>` to create a file") == std::string::npos ||
        custom_tool_description.find("each hunk line has exactly one patch marker") == std::string::npos ||
        custom_tool_description.find("`-` for existing content to remove") == std::string::npos ||
        custom_tool_description.find("the hunk MUST contain the existing content as one or more `-` lines") == std::string::npos ||
        custom_tool_description.find("the replacement content as one or more `+` lines") == std::string::npos ||
        custom_tool_description.find("Context lines are unchanged neighboring lines around that remove/add pair") == std::string::npos ||
        custom_tool_description.find(" environment=prod\n-mode=legacy\n+mode=current\n retries=3") == std::string::npos ||
        custom_tool_description.find("After the single patch marker, the rest of each line is literal file content") == std::string::npos ||
        custom_tool_description.find("+enabled=true\n+timeout=30") == std::string::npos ||
        custom_tool_description.find("intentional append at the end of an existing file") == std::string::npos ||
        custom_tool_description.find("+*** literal content") == std::string::npos ||
        custom_tool_description.find("`+old` followed by `+new`") != std::string::npos ||
        custom_tool_description.find("@@ -10,4 +10,5 @@") != std::string::npos ||
        custom_tool_description.find("`@@ <context>`") != std::string::npos ||
        custom_tool_description.find("`@@ <existing line text>`") != std::string::npos ||
        custom_tool_description.find("Write `@@ `") != std::string::npos ||
        custom_tool_description.find("@@ environment=prod") != std::string::npos ||
        custom_tool_description.find("@@ [server]") != std::string::npos) {
        std::cerr << "Responses apply_patch bridge did not expose replacement-explicit positive Codex patch-format guidance\n";
        return 1;
    }

    const json tool_search_request = {
        {"model", "test-model"},
        {"input", "Find calendar tools"},
        {"tools", json::array({
            {
                {"type", "tool_search"},
                {"execution", "client"},
                {"description", "Search deferred tools"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {{"query", {{"type", "string"}}}}},
                    {"required", json::array({"query"})},
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
    const std::string tool_search_description =
        converted_search.at("tools")[0]["function"].value("description", std::string());
    if (tool_search_description.find("Search deferred tools") == std::string::npos ||
        tool_search_description.find("deferred tool metadata") == std::string::npos ||
        tool_search_description.find("does not perform the underlying task") == std::string::npos ||
        tool_search_description.find("not for the task-specific data or arguments") == std::string::npos ||
        tool_search_description.find("dynamically callable") == std::string::npos ||
        tool_search_description.find("exact returned tool name") == std::string::npos ||
        tool_search_description.find("desired tool or capability") == std::string::npos) {
        std::cerr << "Responses tool_search bridge did not expose deferred-tool lifecycle guidance\n";
        return 1;
    }

    const json followup_request = {
        {"model", "test-model"},
        {"input", json::array({
            {
                {"type", "message"},
                {"role", "user"},
                {"content", json::array({{{"type", "input_text"}, {"text", "Find calendar tools"}}})},
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
                {"tools", json::array({
                    {
                        {"type", "namespace"},
                        {"name", "mcp__calendar"},
                        {"description", "Calendar tools"},
                        {"tools", json::array({
                            {
                                {"type", "function"},
                                {"name", "create_event"},
                                {"description", "Create an event"},
                                {"defer_loading", true},
                                {"parameters", {
                                    {"type", "object"},
                                    {"properties", {{"title", {{"type", "string"}}}}},
                                    {"required", json::array({"title"})},
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
    json followup_tool_context;
    try {
        followup_tool_context = json::parse(followup_messages[2]["content"].get<std::string>());
    } catch (const std::exception & e) {
        std::cerr << "tool_search_output model context was not valid JSON: " << e.what() << "\n";
        return 1;
    }
    if (!followup_tool_context.value("deferred_tools_now_callable", false) ||
        followup_tool_context.value("instruction", std::string()).find("dynamically callable") == std::string::npos ||
        followup_tool_context.value("instruction", std::string()).find("exact returned tool name") == std::string::npos ||
        !followup_tool_context.contains("tools") ||
        !followup_tool_context.at("tools").is_array() ||
        followup_tool_context.at("tools").size() != 1 ||
        followup_tool_context.at("tools")[0]["function"]["name"] != "mcp__calendar.create_event") {
        std::cerr << "tool_search_output did not mark the discovered tool as dynamically callable\n";
        return 1;
    }

    const json standalone_output_request = {
        {"model", "test-model"},
        {"input", json::array({
            {
                {"type", "function_call_output"},
                {"name", "notifications"},
                {"namespace", "slack"},
                {"output", "Alice mentioned you."},
            },
        })},
    };
    const json converted_standalone_output =
        server_chat_convert_responses_to_chatcmpl(standalone_output_request);
    const json & standalone_messages = converted_standalone_output.at("messages");
    if (standalone_messages.size() != 1 ||
        standalone_messages[0].value("role", std::string()) != "tool" ||
        standalone_messages[0].value("name", std::string()) != "slack.notifications" ||
        standalone_messages[0].value("content", std::string()) != "Alice mentioned you." ||
        standalone_messages[0].contains("tool_call_id")) {
        std::cerr << "Standalone named function output was not preserved as unpaired tool context\n";
        return 1;
    }

    json null_call_id_request = standalone_output_request;
    null_call_id_request["input"][0]["call_id"] = nullptr;
    const json converted_null_call_id = server_chat_convert_responses_to_chatcmpl(null_call_id_request);
    const json & null_call_id_messages = converted_null_call_id.at("messages");
    if (null_call_id_messages.size() != 1 ||
        null_call_id_messages[0].value("role", std::string()) != "tool" ||
        null_call_id_messages[0].value("name", std::string()) != "slack.notifications" ||
        null_call_id_messages[0].value("content", std::string()) != "Alice mentioned you." ||
        null_call_id_messages[0].contains("tool_call_id")) {
        std::cerr << "Standalone named function output with null call ID was not preserved as unpaired tool context\n";
        return 1;
    }

    const json paired_output_request = {
        {"model", "test-model"},
        {"input", json::array({
            {
                {"type", "function_call_output"},
                {"call_id", "call_123"},
                {"output", "Paired output"},
            },
        })},
    };
    const json converted_paired_output = server_chat_convert_responses_to_chatcmpl(paired_output_request);
    const json & paired_messages = converted_paired_output.at("messages");
    if (paired_messages.size() != 1 ||
        paired_messages[0].value("role", std::string()) != "tool" ||
        paired_messages[0].value("content", std::string()) != "Paired output" ||
        paired_messages[0].value("tool_call_id", std::string()) != "call_123" ||
        paired_messages[0].contains("name")) {
        std::cerr << "Paired function output did not preserve its call ID\n";
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
        json({{"query", "calendar"}}).dump(),
        "search-1",
    });
    common_chat_msg search_result = message(
        "tool",
        json({{"tools", converted_followup.at("__llamacpp_responses_deferred_tools")}}).dump());
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
        json({{"query", "calendar"}, {"limit", 8}}).dump(),
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
            inputs.chat_template_kwargs["reasoning_effort"] = json(effort).dump();
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
        json({{"command", "pwd"}}).dump(),
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
