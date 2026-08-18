from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Responses phase generation/parser hardening.
# ---------------------------------------------------------------------------
chat_path = Path("common/chat.cpp")
chat = chat_path.read_text()

chat = replace_once(
    chat,
'''    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    const bool json_string_args   = tmpl.source().find("llama.cpp:xml-string-args=json") != std::string::npos;
    const bool responses_phase_protocol =
        tmpl.source().find("llama.cpp:responses-phase=marker-v1") != std::string::npos &&
        inputs.extra_context.is_object() &&
        inputs.extra_context.value("responses_phase_protocol", false);
''',
'''    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    const bool json_string_args   = tmpl.source().find("llama.cpp:xml-string-args=json") != std::string::npos;
    const bool responses_phase_protocol =
        tmpl.source().find("llama.cpp:responses-phase=marker-v1") != std::string::npos &&
        inputs.extra_context.is_object() &&
        inputs.extra_context.value("responses_phase_protocol", false);
    auto include_grammar =
        responses_phase_protocol ||
        has_response_format ||
        (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
''',
    "enable eager Responses phase grammar",
)

chat = replace_once(
    chat,
'''        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
        }

        auto phase = p.eps();
        if (responses_phase_protocol) {
            auto phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("commentary") | p.literal("final_answer")) +
                p.literal("</response_phase>"));
            phase = p.optional(phase_block + p.space());
        }
''',
'''        auto phase = p.eps();
        if (responses_phase_protocol) {
            auto phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("commentary") | p.literal("final_answer")) +
                p.literal("</response_phase>"));
            // New Responses generations must declare their lifecycle phase before
            // any visible content or tool call. Historical Responses items remain
            // tolerant because they enter as structured messages, not through this
            // generated-text parser.
            phase = phase_block + p.space();
        }

        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            if (responses_phase_protocol) {
                // A bare literal "</think>" may legitimately occur inside model
                // reasoning when discussing the protocol itself. Treat only the
                // close-tag + Responses phase opener pair as the reasoning boundary.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until("</think>\\n<response_phase>")) +
                                       p.literal("</think>\\n"));
            } else {
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                       (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
            }
        }
''',
    "require phase and harden reasoning delimiter",
)

chat = replace_once(
    chat,
'''        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
''',
'''        data.grammar_lazy =
            has_tools &&
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO &&
            !responses_phase_protocol;
''',
    "disable lazy grammar for Responses phases",
)

chat_path.write_text(chat)


# ---------------------------------------------------------------------------
# Keep the already-tested prompt clarification and make thinking-disabled
# Responses framing use the same close-tag + phase boundary as the parser.
# ---------------------------------------------------------------------------
template_path = Path("models/templates/llama-cpp-qwen3.8-codex.jinja")
template = template_path.read_text()

template = replace_once(
    template,
'''    {{- '\\n</tools>' }}
    {%- if _tool_format == 'json' %}
        {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1", "example_parameter_2": "This is the value for the second parameter"}}\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: a single JSON object with "name" and "arguments" keys must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}
    {%- else %}
        {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\n"value_1"\\n</parameter>\\n<parameter=example_parameter_2>\\n"This is the value for the second parameter\\\\nthat can span\\\\nmultiple lines"\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- String parameter values MUST use JSON string syntax inside the XML parameter, including surrounding double quotes and JSON escapes such as \\\\n for embedded newlines\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}
    {%- endif %}
''',
'''    {{- '\\n</tools>' }}
    {%- set tool_call_lead =
        'If you choose to call a function ONLY reply in the following format with NO suffix:'
    %}
    {%- set tool_call_prefix = '' %}
    {%- set tool_phase_reminder = '' %}
    {%- if responses_phase_protocol %}
        {%- set tool_call_lead =
            'If you choose to call a function, first emit the required internal Responses phase block and then the function call in the following format, with NO suffix after </tool_call>:'
        %}
        {%- set tool_call_prefix =
            '<response_phase>commentary</response_phase>\\n'
        %}
        {%- set tool_phase_reminder =
            '- Every function-calling assistant message MUST emit exactly one <response_phase>commentary</response_phase> immediately after </think> and before any visible assistant text or tool call\\n- "NO suffix" means no output after </tool_call>; it does not prohibit the required phase block before the tool call\\n'
        %}
    {%- endif %}
    {%- if _tool_format == 'json' %}
        {{- '\\n\\n' ~ tool_call_lead ~ '\\n\\n'
            ~ tool_call_prefix
            ~ '<tool_call>\\n{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1", "example_parameter_2": "This is the value for the second parameter"}}\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n'
            ~ tool_phase_reminder
            ~ '- Function calls MUST follow the specified format: a single JSON object with "name" and "arguments" keys must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>'
        }}
    {%- else %}
        {{- '\\n\\n' ~ tool_call_lead ~ '\\n\\n'
            ~ tool_call_prefix
            ~ '<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\n"value_1"\\n</parameter>\\n<parameter=example_parameter_2>\\n"This is the value for the second parameter\\\\nthat can span\\\\nmultiple lines"\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n'
            ~ tool_phase_reminder
            ~ '- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- String parameter values MUST use JSON string syntax inside the XML parameter, including surrounding double quotes and JSON escapes such as \\\\n for embedded newlines\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>'
        }}
    {%- endif %}
''',
    "phase-aware tool instructions",
)

template = replace_once(
    template,
'''    {%- if not _thinking_enabled %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
''',
'''    {%- if not _thinking_enabled %}
        {%- if responses_phase_protocol %}
            {{- '<think>\\n\\n</think>\\n' }}
        {%- else %}
            {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- endif %}
    {%- else %}
''',
    "thinking-disabled Responses phase boundary",
)

template_path.write_text(template)


# ---------------------------------------------------------------------------
# Regression coverage: eager phase grammar, literal </think> in reasoning,
# strict current generation, tolerant structured history, and no-think framing.
# ---------------------------------------------------------------------------
test_path = Path("tests/test-qwen38-codex-template.cpp")
test = test_path.read_text()

test = replace_once(
    test,
'''    const auto phase_params = common_chat_templates_apply(tmpls.get(), phase_inputs);

    const std::string historical_phase_marker =
''',
'''    const auto phase_params = common_chat_templates_apply(tmpls.get(), phase_inputs);
    if (phase_params.grammar.empty() || phase_params.grammar_lazy) {
        std::cerr << "Responses phase protocol did not enable an eager generation grammar\\n";
        return 1;
    }

    const std::string historical_phase_marker =
''',
    "eager phase grammar regression",
)

test = replace_once(
    test,
'''    // The model signal is preferred, but omission must remain backward
    // compatible: the old Qwen parse still succeeds and leaves phase empty so
    // Responses egress can apply its structural fallback.
    const auto parsed_unmarked = common_chat_parse(
        "Legacy reasoning.\\n</think>\\n\\nLegacy final text.", false, phase_parser);
    if (!parsed_unmarked.phase.empty() || parsed_unmarked.content != "Legacy final text.") {
        std::cerr << "Unmarked Qwen output did not preserve Responses phase fallback\\n";
        return 1;
    }

    auto no_think_inputs = phase_inputs;
''',
'''    // Regression for a real runtime leak: the model may mention the literal
    // </think> delimiter while reasoning about its own protocol. Only the
    // close-tag immediately paired with the Responses phase opener may end
    // reasoning; the earlier literal must remain hidden reasoning text.
    const std::string generated_delimiter_collision =
        "The literal </think> token is being discussed, not emitted as a boundary.\\n"
        "Still reasoning after that literal token.\\n"
        "</think>\\n<response_phase>final_answer</response_phase>\\n"
        "Only this sentence is visible.";
    const auto parsed_delimiter_collision =
        common_chat_parse(generated_delimiter_collision, false, phase_parser);
    if (parsed_delimiter_collision.phase != "final_answer" ||
        parsed_delimiter_collision.content != "Only this sentence is visible." ||
        parsed_delimiter_collision.reasoning_content.find("literal </think> token") == std::string::npos ||
        parsed_delimiter_collision.reasoning_content.find("Still reasoning after") == std::string::npos ||
        parsed_delimiter_collision.content.find("</think>") != std::string::npos ||
        parsed_delimiter_collision.content.find("response_phase") != std::string::npos) {
        std::cerr << "Literal </think> inside reasoning leaked into visible Responses content\\n";
        return 1;
    }

    // Current Responses generations are strict: an unmarked generated message
    // must not become visible content. Missing phase in durable history remains
    // supported separately because history enters as structured messages.
    bool rejected_unmarked_generation = false;
    try {
        const auto parsed_unmarked_generation = common_chat_parse(
            "Legacy reasoning.\\n</think>\\n\\nLegacy final text.", false, phase_parser);
        rejected_unmarked_generation =
            parsed_unmarked_generation.content.empty() &&
            parsed_unmarked_generation.tool_calls.empty();
    } catch (const std::exception &) {
        rejected_unmarked_generation = true;
    }
    if (!rejected_unmarked_generation) {
        std::cerr << "Responses parser accepted visible generated content without a phase marker\\n";
        return 1;
    }

    auto unmarked_history_request = phase_history_request;
    unmarked_history_request["input"][1].erase("phase");
    const auto converted_unmarked_history =
        server_chat_convert_responses_to_chatcmpl(unmarked_history_request);
    if (!converted_unmarked_history.contains("messages") ||
        converted_unmarked_history.at("messages").size() != 2 ||
        converted_unmarked_history.at("messages")[1].contains("phase")) {
        std::cerr << "Responses ingress stopped accepting historical assistant messages without phase\\n";
        return 1;
    }

    auto unmarked_history_inputs = phase_inputs;
    unmarked_history_inputs.messages =
        common_chat_msgs_parse_oaicompat(converted_unmarked_history.at("messages"));
    const auto unmarked_history_params =
        common_chat_templates_apply(tmpls.get(), unmarked_history_inputs);
    if (unmarked_history_params.prompt.find(
            "</think>\\n\\nChecking the repository.") == std::string::npos) {
        std::cerr << "Unmarked historical assistant message was not rendered tolerantly\\n";
        return 1;
    }

    auto no_think_inputs = phase_inputs;
''',
    "reasoning delimiter and strict generation regressions",
)

test = replace_once(
    test,
'''    if (no_think_params.generation_prompt.find("<think>\\n\\n</think>\\n\\n") == std::string::npos) {
        std::cerr << "Qwen thinking-disabled generation prompt lost its empty think framing\\n";
        return 1;
    }
''',
'''    if (no_think_params.generation_prompt.find("<think>\\n\\n</think>\\n") == std::string::npos ||
        no_think_params.generation_prompt.find("<think>\\n\\n</think>\\n\\n") != std::string::npos) {
        std::cerr << "Qwen thinking-disabled Responses prompt lost the strict phase boundary\\n";
        return 1;
    }
''',
    "thinking-disabled phase prompt regression",
)

test_path.write_text(test)


# ---------------------------------------------------------------------------
# Existing low-noise runtime diagnostics.
# ---------------------------------------------------------------------------
server_path = Path("tools/server/server-task.cpp")
server = server_path.read_text()

server = replace_once(
    server,
'''static std::string server_task_response_message_phase(const common_chat_msg & msg) {
    if (msg.phase == "commentary" || msg.phase == "final_answer") {
        return msg.phase;
    }
    return msg.tool_calls.empty() ? "final_answer" : "commentary";
}
''',
'''static std::string server_task_response_message_phase(const common_chat_msg & msg) {
    if (msg.phase == "commentary" || msg.phase == "final_answer") {
        return msg.phase;
    }
    return msg.tool_calls.empty() ? "final_answer" : "commentary";
}

static std::string server_task_response_message_phase_debug(
        const common_chat_msg & msg,
        const char * event) {
    const std::string resolved = server_task_response_message_phase(msg);
    SRV_INF(
        "[responses-phase] event=%s model_phase=%s resolved_phase=%s tool_calls=%zu\\n",
        event,
        msg.phase.empty() ? "<none>" : msg.phase.c_str(),
        resolved.c_str(),
        msg.tool_calls.size());
    return resolved;
}
''',
    "phase helper",
)

server = replace_once(
    server,
'''    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
''',
'''    if (!new_msg.empty()) {
        if (new_msg.phase != msg_prv_copy.phase) {
            SRV_INF(
                "[responses-phase] parser partial=%s previous=%s current=%s generated_chars=%zu\\n",
                is_partial ? "true" : "false",
                msg_prv_copy.phase.empty() ? "<none>" : msg_prv_copy.phase.c_str(),
                new_msg.phase.empty() ? "<none>" : new_msg.phase.c_str(),
                generated_text.size());
        }
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
''',
    "parser phase transition",
)

server = replace_once(
    server,
'''            {"phase",  server_task_response_message_phase(msg)},
''',
'''            {"phase",  server_task_response_message_phase_debug(msg, "response.completed.output")},
''',
    "non-stream phase",
)

server = replace_once(
    server,
'''            {"phase",   server_task_response_message_phase(oaicompat_msg)}
''',
'''            {"phase",   server_task_response_message_phase_debug(oaicompat_msg, "response.output_item.done")}
''',
    "stream done phase",
)

server = replace_once(
    server,
'''                if (!oai_resp_message_phase.empty()) {
                    message_item["phase"] = oai_resp_message_phase;
                }
                events.push_back(json {
''',
'''                if (!oai_resp_message_phase.empty()) {
                    message_item["phase"] = oai_resp_message_phase;
                }
                SRV_INF(
                    "[responses-phase] event=response.output_item.added model_phase=%s emitted_phase=%s\\n",
                    oai_resp_message_phase.empty() ? "<none>" : oai_resp_message_phase.c_str(),
                    oai_resp_message_phase.empty() ? "<none>" : oai_resp_message_phase.c_str());
                events.push_back(json {
''',
    "stream added phase",
)

server_path.write_text(server)
