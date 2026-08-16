#include "chat.h"

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
        R"(find /home/hausen/src -name "qwen38-llamacpp-research.md" 2>/dev/null)";
    const std::string shell_arguments = nlohmann::ordered_json({
        {"command", shell_command},
        {"workdir", "/home/hausen/src"},
    }).dump();

    const common_chat_msg system = message("system", "Stable Codex instructions.");
    const common_chat_msg user = message("user", "Initial user request.");

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

    const std::string rendered_command =
        "<parameter=command>\n" + shell_command + "\n</parameter>";
    if (before.prompt.find(rendered_command) == std::string::npos) {
        std::cerr << "Responses JSON-string arguments were not normalized and rendered losslessly as XML parameters\n";
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
        "<parameter=command>\n" + shell_command + "\n</parameter>\n"
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

    std::cout << "Qwen3.8 Codex tool format and compaction prefix tests passed\n";
    return 0;
}
