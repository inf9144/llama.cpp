from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one marker in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1))


# common/chat.cpp: fix the phase flag lookup and keep phase structurally
# separate from reasoning in the Qwen3-Coder/Qwen3.8 specialized parser.
replace_once(
    "common/chat.cpp",
    '''    const bool responses_phase_protocol = inputs.extra_context.has_value() &&
        inputs.extra_context->value("responses_phase_protocol", false);
''',
    '''    const bool responses_phase_protocol = inputs.extra_context.is_object() &&
        inputs.extra_context.value("responses_phase_protocol", false);
''',
)
replace_once(
    "common/chat.cpp",
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

        // Response format parser
        if (has_response_format) {
            return generation_prompt + (reasoning << p.content(p.schema(p.json(), "response-format", inputs.json_schema)));
        }
''',
    '''        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
        }

        auto phase = p.eps();
        if (responses_phase_protocol) {
            phase = p.literal("<response_phase>") +
                    p.phase(p.literal("commentary") | p.literal("final_answer")) +
                    p.literal("</response_phase>") + p.space();
        }

        // Response format parser
        if (has_response_format) {
            return generation_prompt + (reasoning << phase << p.content(p.schema(p.json(), "response-format", inputs.json_schema)));
        }
''',
)
replace_once(
    "common/chat.cpp",
    '''            return generation_prompt +
                   (reasoning << p.content(p.until_one_of(tool_call_starts)) << tool_calls);
        }

        // Content only parser
        return generation_prompt + (reasoning << p.content(p.rest()));
    });
''',
    '''            return generation_prompt +
                   (reasoning << phase << p.content(p.until_one_of(tool_call_starts)) << tool_calls);
        }

        // Content only parser
        return generation_prompt + (reasoning << phase << p.content(p.rest()));
    });
''',
)

# Jinja: phase is its own internal block after reasoning, never part of <think>.
replace_once(
    "models/templates/llama-cpp-qwen3.8-codex.jinja",
    '''        'Responses message phases are transport metadata. When you generate a <think> block, immediately before closing </think> emit exactly one internal phase marker: <response_phase>commentary</response_phase> when the visible assistant message is an intermediate update before tool use or further work, or <response_phase>final_answer</response_phase> when the visible assistant message is the terminal answer of the turn. The marker must stay inside <think>; never repeat or mention it in visible text.'
''',
    '''        'Responses message phases are transport metadata and are separate from reasoning. Emit exactly one internal phase block immediately after </think> and before any visible assistant text or tool call: <response_phase>commentary</response_phase> when the visible assistant message is an intermediate update before tool use or further work, or <response_phase>final_answer</response_phase> when the visible assistant message is the terminal answer of the turn. If the template has already supplied an empty <think></think> block, begin your generation with the phase block. Never repeat or mention the phase block in visible text.'
''',
)
replace_once(
    "models/templates/llama-cpp-qwen3.8-codex.jinja",
    '''        {%- set phase_marker = '' %}
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
    '''        {%- set phase_prefix = '' %}
        {%- set after_think = '\\n\\n' %}
        {%- if responses_phase_protocol
               and message.phase is defined
               and message.phase is not none
               and message.phase %}
            {%- if message.phase not in ('commentary', 'final_answer') %}
                {{- raise_exception('Unexpected Responses message phase ' ~ message.phase ~ '.') }}
            {%- endif %}
            {%- set phase_prefix =
                '<response_phase>' ~ message.phase ~ '</response_phase>\\n'
            %}
            {%- set after_think = '\\n' ~ phase_prefix %}
        {%- endif %}
        {#- Preserve the original invariant: when thinking is preserved,
            always reconstruct the thinking block, even when it is empty. -#}
        {%- if _preserve_thinking
               or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>assistant\\n<think>\\n'
                ~ reasoning_content
                ~ '\\n</think>'
                ~ after_think
                ~ content
            }}
        {%- else %}
            {{- '<|im_start|>assistant\\n' ~ phase_prefix ~ content }}
        {%- endif %}
''',
)

# Focused regressions: assert separate phase block, atomic partial parsing, and
# independence from model-generated reasoning.
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    const std::string historical_phase_marker =
        "<response_phase>commentary</response_phase>\\n</think>\\n\\nChecking the repository.";
''',
    '''    const std::string historical_phase_marker =
        "</think>\\n<response_phase>commentary</response_phase>\\nChecking the repository.";
''',
)
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    const std::string generated_commentary =
        "I should inspect the workspace.\\n" + commentary_marker + "\\n</think>\\n\\n"
        "I'll inspect the repository.\\n\\n"
''',
    '''    const std::string generated_commentary =
        "I should inspect the workspace.\\n</think>\\n" + commentary_marker + "\\n"
        "I'll inspect the repository.\\n\\n"
''',
)
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''        const auto partial_phase = common_chat_parse(
            "I should inspect the workspace.\\n" + commentary_marker.substr(0, n), true, phase_parser);
''',
    '''        const auto partial_phase = common_chat_parse(
            "I should inspect the workspace.\\n</think>\\n" + commentary_marker.substr(0, n), true, phase_parser);
''',
)
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    const std::string generated_final =
        "The task is complete.\\n<response_phase>final_answer</response_phase>\\n</think>\\n\\nDone.";
''',
    '''    const std::string generated_final =
        "The task is complete.\\n</think>\\n<response_phase>final_answer</response_phase>\\nDone.";
''',
)
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    server_task_result_cmpl_final final_phase_result;
''',
    '''    auto no_think_inputs = phase_inputs;
    no_think_inputs.messages = { system, user };
    no_think_inputs.enable_thinking = false;
    const auto no_think_params = common_chat_templates_apply(tmpls.get(), no_think_inputs);
    if (no_think_params.generation_prompt.find("<think>\\n\\n</think>\\n\\n") == std::string::npos) {
        std::cerr << "Qwen thinking-disabled generation prompt lost its empty think framing\\n";
        return 1;
    }
    common_chat_parser_params no_think_parser(no_think_params);
    no_think_parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    no_think_parser.parser.load(no_think_params.parser);
    const auto parsed_no_think = common_chat_parse(
        "<response_phase>final_answer</response_phase>\\nDone without thinking.", false, no_think_parser);
    if (parsed_no_think.phase != "final_answer" || parsed_no_think.content != "Done without thinking.") {
        std::cerr << "Qwen phase protocol depended on model-generated reasoning\\n";
        return 1;
    }

    server_task_result_cmpl_final final_phase_result;
''',
)
