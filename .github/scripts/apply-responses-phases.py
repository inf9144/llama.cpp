from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one marker in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# Generic parsed-message metadata. Keep phase internal: Chat Completions
# serialization remains unchanged; Responses egress consumes it explicitly.
# ---------------------------------------------------------------------------
replace_once(
    "common/chat.h",
    '''    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               tool_name;
    std::string                               tool_call_id;
''',
    '''    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               phase;
    std::string                               tool_name;
    std::string                               tool_call_id;
''',
)
replace_once(
    "common/chat.h",
    '''        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() &&
               tool_name.empty() && tool_call_id.empty();
''',
    '''        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() &&
               phase.empty() && tool_name.empty() && tool_call_id.empty();
''',
)
replace_once(
    "common/chat.h",
    '''        return role == other.role && content == other.content && content_parts == other.content_parts &&
               tool_calls == other.tool_calls && reasoning_content == other.reasoning_content &&
               tool_name == other.tool_name && tool_call_id == other.tool_call_id;
''',
    '''        return role == other.role && content == other.content && content_parts == other.content_parts &&
               tool_calls == other.tool_calls && reasoning_content == other.reasoning_content &&
               phase == other.phase && tool_name == other.tool_name && tool_call_id == other.tool_call_id;
''',
)

replace_once(
    "common/chat-peg-parser.h",
    '''    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING       = "reasoning";
    static constexpr const char * CONTENT         = "content";
''',
    '''    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING       = "reasoning";
    static constexpr const char * CONTENT         = "content";
    static constexpr const char * PHASE           = "phase";
''',
)
replace_once(
    "common/chat-peg-parser.h",
    '''    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }

    common_peg_parser tag_with_safe_content''',
    '''    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }

    common_peg_parser phase(const common_peg_parser & p) { return atomic(tag(PHASE, p)); }

    common_peg_parser tag_with_safe_content''',
)
replace_once(
    "common/chat-peg-parser.cpp",
    '''    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;
''',
    '''    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;
    bool is_phase     = node.tag == common_chat_peg_builder::PHASE;
''',
)
replace_once(
    "common/chat-peg-parser.cpp",
    '''    if (is_content) {
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += std::string(node.text);
    }

    // Handle tool-related tags''',
    '''    if (is_content) {
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += std::string(node.text);
    }

    if (is_phase) {
        result.phase = std::string(node.text);
    }

    // Handle tool-related tags''',
)

# Parse phase from Responses-derived assistant history, but do not expose it from
# common_chat_msg::to_json_oaicompat(). The Jinja renderer receives it only when
# the Responses phase protocol flag is active.
replace_once(
    "common/chat.cpp",
    '''            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
''',
    '''            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("phase") && !message.at("phase").is_null()) {
                if (!message.at("phase").is_string()) {
                    throw std::invalid_argument("Invalid 'phase' type: expected string");
                }
                msg.phase = message.at("phase");
            }
            if (message.contains("name")) {
''',
)
replace_once(
    "common/chat.cpp",
    '''static json render_message_to_json(const std::vector<common_chat_msg> & msgs, const jinja::caps & c) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\\n", __func__);
    }

    json messages = json::array();
    for (const auto & msg : msgs) {
        messages.push_back(msg.to_json_oaicompat(/* concat_typed_text= */ false));
    }
    return messages_inp_normalizer(c).normalize(messages);
}
''',
    '''static json render_message_to_json(
        const std::vector<common_chat_msg> & msgs,
        const jinja::caps & c,
        bool include_phase = false) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\\n", __func__);
    }

    json messages = json::array();
    for (const auto & msg : msgs) {
        json rendered = msg.to_json_oaicompat(/* concat_typed_text= */ false);
        if (include_phase && !msg.phase.empty()) {
            rendered["phase"] = msg.phase;
        }
        messages.push_back(std::move(rendered));
    }
    return messages_inp_normalizer(c).normalize(messages);
}
''',
)
replace_once(
    "common/chat.cpp",
    '''    params.messages              = render_message_to_json(*messages_to_render, tmpl.original_caps());
    params.tool_choice           = inputs.tool_choice;
''',
    '''    const auto phase_protocol_it = inputs.chat_template_kwargs.find("responses_phase_protocol");
    const bool responses_phase_protocol =
        phase_protocol_it != inputs.chat_template_kwargs.end() && phase_protocol_it->second == "true";
    params.messages              = render_message_to_json(
        *messages_to_render, tmpl.original_caps(), responses_phase_protocol);
    params.tool_choice           = inputs.tool_choice;
''',
)

# Qwen3-Coder/Qwen3.8: phase metadata lives inside the generated <think> block.
# This reuses the existing stop-aware reasoning parser, so phase markers never
# become visible content or reasoning deltas.
replace_once(
    "common/chat.cpp",
    '''    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    const bool json_string_args   = tmpl.source().find("llama.cpp:xml-string-args=json") != std::string::npos;
''',
    '''    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    const bool json_string_args   = tmpl.source().find("llama.cpp:xml-string-args=json") != std::string::npos;
    const bool responses_phase_protocol = inputs.extra_context.has_value() &&
        inputs.extra_context->value("responses_phase_protocol", false);
''',
)
replace_once(
    "common/chat.cpp",
    '''        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
        }
''',
    '''        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            if (responses_phase_protocol) {
                auto phase_marker = p.literal("<response_phase>") +
                    p.phase(p.literal("commentary") | p.literal("final_answer")) +
                    p.literal("</response_phase>");
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({ "<response_phase>", "</think>", "<tool_call>" })) +
                                       p.optional(phase_marker + p.space()) +
                                       (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
            } else {
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                       (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
            }
        }
''',
)

# ---------------------------------------------------------------------------
# Qwen Codex Jinja: explain the transport metadata and replay historical phases
# inside <think>, never in visible assistant text.
# ---------------------------------------------------------------------------
replace_once(
    "models/templates/llama-cpp-qwen3.8-codex.jinja",
    '''{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}
{%- set auto_disable_thinking_with_tools =
''',
    '''{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}
{%- set responses_phase_protocol =
    responses_phase_protocol if responses_phase_protocol is defined else false
%}
{%- set auto_disable_thinking_with_tools =
''',
)
replace_once(
    "models/templates/llama-cpp-qwen3.8-codex.jinja",
    '''    {%- endif %}
{%- endif %}
{#- -------------------------------------------------------------------------
    Content rendering
''',
    '''    {%- endif %}
{%- endif %}
{%- if responses_phase_protocol %}
    {%- set phase_instructions =
        'Responses message phases are transport metadata. When you generate a <think> block, immediately before closing </think> emit exactly one internal phase marker: <response_phase>commentary</response_phase> when the visible assistant message is an intermediate update before tool use or further work, or <response_phase>final_answer</response_phase> when the visible assistant message is the terminal answer of the turn. The marker must stay inside <think>; never repeat or mention it in visible text.'
    %}
    {%- if reasoning_instructions %}
        {%- set reasoning_instructions =
            reasoning_instructions ~ '\n\n' ~ phase_instructions
        %}
    {%- else %}
        {%- set reasoning_instructions = phase_instructions %}
    {%- endif %}
{%- endif %}
{#- -------------------------------------------------------------------------
    Content rendering
''',
)
replace_once(
    "models/templates/llama-cpp-qwen3.8-codex.jinja",
    '''        {%- set reasoning_content = reasoning_content | trim %}
        {#- Preserve the original invariant: when thinking is preserved,
            always reconstruct the thinking block, even when it is empty. -#}
        {%- if _preserve_thinking
               or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>assistant\\n<think>\\n'
                ~ reasoning_content
                ~ '\\n</think>\\n\\n'
                ~ content
            }}
        {%- else %}
            {{- '<|im_start|>assistant\\n' ~ content }}
        {%- endif %}
''',
    '''        {%- set reasoning_content = reasoning_content | trim %}
        {%- set phase_marker = '' %}
        {%- if responses_phase_protocol
               and message.phase is defined
               and message.phase is not none
               and message.phase %}
            {%- if message.phase not in ('commentary', 'final_answer') %}
                {{- raise_exception('Unexpected Responses message phase ' ~ message.phase ~ '.') }}
            {%- endif %}
            {%- set phase_marker =
                '\\n<response_phase>' ~ message.phase ~ '</response_phase>'
            %}
        {%- endif %}
        {#- Preserve the original invariant: when thinking is preserved,
            always reconstruct the thinking block, even when it is empty. -#}
        {%- if _preserve_thinking
               or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>assistant\\n<think>\\n'
                ~ reasoning_content
                ~ phase_marker
                ~ '\\n</think>\\n\\n'
                ~ content
            }}
        {%- else %}
            {{- '<|im_start|>assistant\\n' ~ content }}
        {%- endif %}
''',
)

# ---------------------------------------------------------------------------
# Responses ingress: preserve phase on internal assistant history and opt the
# chat template into the Qwen phase protocol. This flag is server-internal.
# ---------------------------------------------------------------------------
replace_once(
    "tools/server/server-chat.cpp",
    '''                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
''',
    '''                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (exists_and_is_string(item, "phase")) {
                        prev_msg["phase"] = item.at("phase");
                    }
                    if (!exists_and_is_array(prev_msg, "content")) {
''',
)
replace_once(
    "tools/server/server-chat.cpp",
    '''    if (is_compaction_request) {
        json chat_template_kwargs = json_value(chatcmpl_body, "chat_template_kwargs", json::object());
        if (!chat_template_kwargs.is_object()) {
            throw std::invalid_argument("'chat_template_kwargs' must be an object");
        }

        // Keep model-specific compaction prompting in the chat template. This
        // flag does not alter durable history; a template can append its own
        // compaction tail after rendering the normal message/tool prefix.
        chat_template_kwargs["is_compaction"] = true;
        chatcmpl_body["chat_template_kwargs"] = std::move(chat_template_kwargs);
        chatcmpl_body["__llamacpp_responses_compaction"] = true;
    }
''',
    '''    {
        json chat_template_kwargs = json_value(chatcmpl_body, "chat_template_kwargs", json::object());
        if (!chat_template_kwargs.is_object()) {
            throw std::invalid_argument("'chat_template_kwargs' must be an object");
        }

        // Responses phase metadata is model-assisted by templates that know how
        // to signal it; templates that do not recognize the flag simply ignore it.
        chat_template_kwargs["responses_phase_protocol"] = true;

        if (is_compaction_request) {
            // Keep model-specific compaction prompting in the chat template. This
            // flag does not alter durable history; a template can append its own
            // compaction tail after rendering the normal message/tool prefix.
            chat_template_kwargs["is_compaction"] = true;
        }
        chatcmpl_body["chat_template_kwargs"] = std::move(chat_template_kwargs);
    }

    if (is_compaction_request) {
        chatcmpl_body["__llamacpp_responses_compaction"] = true;
    }
''',
)

# ---------------------------------------------------------------------------
# Responses egress. Explicit model phase wins; legacy/unmarked generations use
# the structural tool-call fallback.
# ---------------------------------------------------------------------------
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

static std::string server_task_response_message_phase(const common_chat_msg & msg) {
    if (msg.phase == "commentary" || msg.phase == "final_answer") {
        return msg.phase;
    }
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
    '''        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };
''',
    '''        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"},
            {"phase",   server_task_response_message_phase(oaicompat_msg)}
        };
''',
)
replace_once(
    "tools/server/server-task.h",
    '''    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;
''',
    '''    std::string oai_resp_message_id;
    std::string oai_resp_message_phase;
    std::string oai_resp_fc_id;
''',
)
replace_once(
    "tools/server/server-task.cpp",
    '''    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id             = state.oai_resp_fc_id;
''',
    '''    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_message_phase = state.chat_msg.phase;
    oai_resp_fc_id             = state.oai_resp_fc_id;
''',
)
replace_once(
    "tools/server/server-task.cpp",
    '''            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
''',
    '''            if (!text_block_started) {
                json message_item = {
                    {"content", json::array()},
                    {"id",      oai_resp_message_id},
                    {"role",    "assistant"},
                    {"status",  "in_progress"},
                    {"type",    "message"},
                };
                if (!oai_resp_message_phase.empty()) {
                    message_item["phase"] = oai_resp_message_phase;
                }
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", std::move(message_item)},
                    }},
                });
''',
)

# ---------------------------------------------------------------------------
# Focused Qwen/Codex regression coverage.
# ---------------------------------------------------------------------------
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

    // Responses phase is internal metadata: it must survive Responses history ->
    // template rendering, but never leak through normal Chat Completions JSON.
    common_chat_msg chat_only_phase = message("assistant", "Internal metadata test.");
    chat_only_phase.phase = "commentary";
    if (chat_only_phase.to_json_oaicompat().contains("phase")) {
        std::cerr << "Responses phase leaked through Chat Completions serialization\\n";
        return 1;
    }

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
    if (!converted_phase_history.contains("chat_template_kwargs") ||
        !converted_phase_history.at("chat_template_kwargs").value("responses_phase_protocol", false) ||
        !converted_phase_history.contains("messages") ||
        converted_phase_history.at("messages").size() != 1 ||
        converted_phase_history.at("messages")[0].value("phase", std::string()) != "commentary") {
        std::cerr << "Responses phase metadata was not preserved for the internal template bridge\\n";
        return 1;
    }

    common_chat_templates_inputs phase_inputs;
    phase_inputs.messages = common_chat_msgs_parse_oaicompat(converted_phase_history.at("messages"));
    phase_inputs.tools = { shell_tool };
    phase_inputs.add_generation_prompt = true;
    phase_inputs.enable_thinking = true;
    phase_inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    phase_inputs.chat_template_kwargs["responses_phase_protocol"] = "true";
    const auto phase_params = common_chat_templates_apply(tmpls.get(), phase_inputs);

    const std::string historical_phase_marker =
        "<response_phase>commentary</response_phase>\\n</think>\\n\\nChecking the repository.";
    if (phase_params.prompt.find(historical_phase_marker) == std::string::npos ||
        phase_params.prompt.find("Responses message phases are transport metadata") == std::string::npos) {
        std::cerr << "Qwen phase protocol was not taught/replayed through the Codex template\\n";
        return 1;
    }

    common_chat_parser_params phase_parser(phase_params);
    phase_parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    phase_parser.parser.load(phase_params.parser);

    const std::string commentary_marker = "<response_phase>commentary</response_phase>";
    const std::string generated_commentary =
        "I should inspect the workspace.\\n" + commentary_marker + "\\n</think>\\n\\n"
        "I'll inspect the repository.\\n\\n"
        "<tool_call>\\n"
        "<function=shell_command>\\n"
        "<parameter=command>\\n\"pwd\"\\n</parameter>\\n"
        "</function>\\n"
        "</tool_call>";
    const auto parsed_commentary = common_chat_parse(generated_commentary, false, phase_parser);
    if (parsed_commentary.phase != "commentary" ||
        parsed_commentary.content.find("response_phase") != std::string::npos ||
        parsed_commentary.reasoning_content.find("response_phase") != std::string::npos ||
        parsed_commentary.tool_calls.size() != 1) {
        std::cerr << "Qwen commentary phase marker was not consumed as metadata\\n";
        return 1;
    }

    // Exercise partial parsing of the internal marker. No marker prefix may
    // escape into a streamed reasoning/content delta while it is incomplete.
    for (size_t n = 1; n < commentary_marker.size(); ++n) {
        const auto partial_phase = common_chat_parse(
            "I should inspect the workspace.\\n" + commentary_marker.substr(0, n), true, phase_parser);
        if (partial_phase.content.find("response_phase") != std::string::npos ||
            partial_phase.reasoning_content.find("response_phase") != std::string::npos) {
            std::cerr << "Partial Qwen phase marker leaked into streamed model text\\n";
            return 1;
        }
    }

    const std::string generated_final =
        "The task is complete.\\n<response_phase>final_answer</response_phase>\\n</think>\\n\\nDone.";
    const auto parsed_final = common_chat_parse(generated_final, false, phase_parser);
    if (parsed_final.phase != "final_answer" || parsed_final.content != "Done." ||
        parsed_final.reasoning_content.find("response_phase") != std::string::npos) {
        std::cerr << "Qwen final_answer phase marker was not consumed as metadata\\n";
        return 1;
    }

    server_task_result_cmpl_final final_phase_result;
    final_phase_result.oaicompat_model = "test-model";
    final_phase_result.oai_resp_id = "resp_final_phase_test";
    final_phase_result.oai_resp_message_id = "msg_final_phase_test";
    final_phase_result.n_prompt_tokens = 0;
    final_phase_result.n_prompt_tokens_cache = 0;
    final_phase_result.n_decoded = 0;
    final_phase_result.oaicompat_msg = parsed_final;
    final_phase_result.oaicompat_msg.role = "assistant";

    const auto final_phase_response = final_phase_result.to_json_oaicompat_resp();
    if (!final_phase_response.contains("output") || final_phase_response.at("output").size() != 1 ||
        final_phase_response.at("output")[0].value("type", std::string()) != "message" ||
        final_phase_response.at("output")[0].value("phase", std::string()) != "final_answer") {
        std::cerr << "Responses egress lost the model-selected final_answer phase\\n";
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
        std::cerr << "Streaming Responses egress lost final_answer phase\\n";
        return 1;
    }

    server_task_result_cmpl_final commentary_phase_result;
    commentary_phase_result.oaicompat_model = "test-model";
    commentary_phase_result.oaicompat_msg = parsed_commentary;
    commentary_phase_result.oaicompat_msg.role = "assistant";
    const auto commentary_phase_response = commentary_phase_result.to_json_oaicompat_resp();
    if (!commentary_phase_response.contains("output") || commentary_phase_response.at("output").size() != 2 ||
        commentary_phase_response.at("output")[0].value("type", std::string()) != "message" ||
        commentary_phase_response.at("output")[0].value("phase", std::string()) != "commentary" ||
        commentary_phase_response.at("output")[1].value("type", std::string()) != "function_call") {
        std::cerr << "Responses egress lost the model-selected commentary phase\\n";
        return 1;
    }

    // Backward compatibility: an unmarked parser result still gets the old
    // structural fallback, so non-participating templates/models keep working.
    common_chat_msg fallback_msg = message("assistant", "Legacy preamble.");
    fallback_msg.tool_calls.push_back({
        "shell_command",
        nlohmann::ordered_json({{"command", "pwd"}}).dump(),
        "fallback_tool",
    });
    server_task_result_cmpl_final fallback_result;
    fallback_result.oaicompat_model = "test-model";
    fallback_result.oaicompat_msg = fallback_msg;
    const auto fallback_response = fallback_result.to_json_oaicompat_resp();
    if (fallback_response.at("output")[0].value("phase", std::string()) != "commentary") {
        std::cerr << "Unmarked Responses message lost the tool-call phase fallback\\n";
        return 1;
    }

    std::cout << "Qwen3.8 Codex user text, generic string framing, custom framing, compaction, tool search, and Responses phase tests passed\\n";
''',
)
