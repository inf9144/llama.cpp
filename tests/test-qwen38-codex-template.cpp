#include "chat.h"

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

    common_chat_tool tool;
    tool.name = "lookup";
    tool.description = "Look up a value";
    tool.parameters = R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})";

    const common_chat_msg system = message("system", "Stable Codex instructions.");
    const common_chat_msg user = message("user", "Initial user request.");

    common_chat_msg assistant_tool_call;
    assistant_tool_call.role = "assistant";
    assistant_tool_call.tool_calls.push_back({
        "lookup",
        R"({"query":"before-compaction"})",
        "call_1",
    });

    common_chat_msg tool_result = message("tool", "result");
    tool_result.tool_name = "lookup";
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
        inputs.tools = { tool };
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = true;
        return common_chat_templates_apply(tmpls.get(), inputs).prompt;
    };

    const std::string before_prompt = render(before_compaction);
    const std::string after_prompt = render(after_compaction);

    const std::string before_prefix = prefix_before_first_user(before_prompt);
    const std::string after_prefix = prefix_before_first_user(after_prompt);

    if (before_prefix != after_prefix) {
        std::cerr << "Codex system/tools prefix changed after tool-call history removal\n";
        return 1;
    }

    const std::string json_tool_instruction =
        "{\"name\": \"example_function_name\", \"arguments\":";
    if (before_prefix.find(json_tool_instruction) == std::string::npos) {
        std::cerr << "Codex template did not default to JSON tool-call instructions\n";
        return 1;
    }

    std::cout << "Qwen3.8 Codex prefix stability test passed\n";
    return 0;
}
