from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_in_qwen(text: str, old: str, new: str, label: str) -> str:
    marker = "static common_chat_params common_chat_params_init_qwen3_coder("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"{label}: qwen3 coder function not found")
    end = text.find("\nstatic common_chat_params ", start + len(marker))
    if end < 0:
        raise SystemExit(f"{label}: next chat parser function not found")
    section = text[start:end]
    count = section.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one qwen match, found {count}")
    section = section.replace(old, new, 1)
    return text[:start] + section + text[end:]


def replace_between(text: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f"{label}: start marker not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"{label}: end marker not found")
    return text[:start] + replacement + text[end:]


# ---------------------------------------------------------------------------
# Carry one explicit parser capability from template selection into the server
# streaming state.  This mode is intentionally narrow: only the Qwen Responses
# bridge uses it.
# ---------------------------------------------------------------------------
chat_h_path = Path("common/chat.h")
chat_h = chat_h_path.read_text()
chat_h = replace_once(
    chat_h,
'''    std::string                         parser;
    common_chat_msg_delimiters          message_delimiters;
};
''',
'''    std::string                         parser;
    common_chat_msg_delimiters          message_delimiters;
    // Text-delimited reasoning cannot be committed safely while streaming when
    // the reasoning itself may quote its closing delimiter.  Templates opting
    // into this mode are parsed only after the complete generated item exists.
    bool                                buffered_reasoning_boundary = false;
};
''',
    "chat params buffered reasoning flag",
)
chat_h = replace_once(
    chat_h,
'''    bool                    debug                = false;  // Enable debug output for PEG parser
    common_peg_arena        parser               = {};
''',
'''    bool                    debug                = false;  // Enable debug output for PEG parser
    bool                    buffered_reasoning_boundary = false;
    common_peg_arena        parser               = {};
''',
    "parser params buffered reasoning flag",
)
chat_h = replace_once(
    chat_h,
'''    common_chat_parser_params(const common_chat_params & chat_params) {
        format  = chat_params.format;
        generation_prompt = chat_params.generation_prompt;
    }
''',
'''    common_chat_parser_params(const common_chat_params & chat_params) {
        format  = chat_params.format;
        generation_prompt = chat_params.generation_prompt;
        buffered_reasoning_boundary = chat_params.buffered_reasoning_boundary;
    }
''',
    "copy buffered reasoning flag",
)
chat_h_path.write_text(chat_h)


# ---------------------------------------------------------------------------
# Qwen Responses v5:
# - phase is no longer model text; server egress derives it from the completed
#   assistant item (tool call => commentary, otherwise final_answer),
# - phase protocol no longer makes the generation grammar eager,
# - parsing uses the ordinary native Qwen </think> syntax; the generic final
#   parse wrapper below resolves quoted delimiters before PEG sees them.
# ---------------------------------------------------------------------------
chat_path = Path("common/chat.cpp")
chat = chat_path.read_text()
chat = replace_in_qwen(
    chat,
'''    auto include_grammar =
        responses_phase_protocol ||
        has_response_format ||
        (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
''',
'''    auto include_grammar =
        has_response_format ||
        (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    data.buffered_reasoning_boundary =
        responses_phase_protocol && supports_reasoning && extract_reasoning;
''',
    "Responses no longer forces eager grammar",
)

chat = replace_in_qwen(
    chat,
'''    auto build_parser = [&](bool strict_phase_generation) {
        return build_chat_peg_parser([&, strict_phase_generation](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        auto phase = p.eps();
        if (responses_phase_protocol) {
            auto phase_block = p.atomic(
                p.literal("<response_phase>") +
                p.phase(p.literal("commentary") | p.literal("final_answer")) +
                p.literal("</response_phase>"));
            // Runtime parsing stays tolerant for partial/legacy material, while
            // the separate generation grammar requires the phase structurally.
            phase = strict_phase_generation
                ? phase_block + p.space()
                : p.optional(phase_block + p.space());
        }

        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            if (responses_phase_protocol && !strict_phase_generation) {
                // Runtime parser hardening only: pair the real reasoning close
                // with the following phase opener, but accept whitespace forms
                // that earlier strict grammars could legally generate.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({
                                           "</think><response_phase>",
                                           "</think> <response_phase>",
                                           "</think>\\t<response_phase>",
                                           "</think>\\n<response_phase>",
                                           "</think>\\n\\n<response_phase>",
                                       })) +
                                       p.literal("</think>") + p.space());
            } else if (responses_phase_protocol && strict_phase_generation) {
                // Fresh Responses generation uses Qwen's native </think> close,
                // followed by one canonical newline before the required phase.
                // Matching </think>\\n also avoids treating an inline literal
                // </think> mention as the reasoning boundary.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until("</think>\\n")) +
                                       p.literal("</think>\\n"));
            } else {
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                       (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
            }
        }
''',
'''    auto build_parser = [&](bool) {
        return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        // Responses phase is transport metadata and is resolved by server egress
        // from the completed assistant item.  Keeping it out of model text also
        // removes an entire delimiter-collision surface from hidden reasoning.
        auto phase = p.eps();

        auto reasoning = p.eps();
        if (supports_reasoning && extract_reasoning) {
            reasoning = p.optional("<think>" + p.space() +
                                   p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                   (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
        }
''',
    "remove model phase parser and restore native Qwen reasoning",
)

chat = replace_in_qwen(
    chat,
'''        data.grammar_lazy =
            has_tools &&
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO &&
            !responses_phase_protocol;

        auto grammar_parser = build_parser(responses_phase_protocol);
''',
'''        data.grammar_lazy =
            has_tools &&
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        auto grammar_parser = build_parser(false);
''',
    "restore lazy Qwen tool grammar",
)

# Final full parse: choose the last unescaped native Qwen closing tag and mask
# earlier literal occurrences before feeding the text to the ordinary PEG.  This
# mirrors Qwen's own full-generation split strategy while leaving visible text
# untouched.  An odd number of backslashes immediately before '<' marks quoted
# data and is never selected as the structural close.
chat = replace_once(
    chat,
'''common_chat_msg common_chat_parse(const std::string &               input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    return common_chat_peg_parse(params.parser, input, is_partial, params);
}
''',
'''static std::optional<size_t> common_chat_last_unescaped_marker(
        const std::string & input,
        const std::string & marker) {
    if (marker.empty() || input.size() < marker.size()) {
        return std::nullopt;
    }
    size_t pos = input.rfind(marker);
    while (pos != std::string::npos) {
        size_t slash_count = 0;
        for (size_t i = pos; i > 0 && input[i - 1] == '\\\\'; --i) {
            ++slash_count;
        }
        if ((slash_count % 2) == 0) {
            return pos;
        }
        if (pos == 0) {
            break;
        }
        pos = input.rfind(marker, pos - 1);
    }
    return std::nullopt;
}

common_chat_msg common_chat_parse(const std::string &               input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    if (!params.buffered_reasoning_boundary || is_partial) {
        return common_chat_peg_parse(params.parser, input, is_partial, params);
    }

    static const std::string think_end = "</think>";
    const auto boundary = common_chat_last_unescaped_marker(input, think_end);
    if (!boundary.has_value()) {
        // Keep the legacy Qwen fallback where a tool call can end reasoning
        // without an explicit </think>.
        return common_chat_peg_parse(params.parser, input, false, params);
    }

    std::string parse_input = input;
    size_t pos = parse_input.find(think_end);
    while (pos != std::string::npos && pos < *boundary) {
        // Preserve byte offsets while making earlier literal tags invisible to
        // the PEG delimiter matcher. The original reasoning bytes are restored
        // below and never exposed as visible content.
        parse_input[pos] = '~';
        pos = parse_input.find(think_end, pos + think_end.size());
    }

    auto msg = common_chat_peg_parse(params.parser, parse_input, false, params);
    msg.reasoning_content = input.substr(0, *boundary);
    return msg;
}
''',
    "buffered final reasoning boundary resolver",
)
chat_path.write_text(chat)


# ---------------------------------------------------------------------------
# Stop teaching or replaying the internal phase marker to Qwen.  The server
# still passes responses_phase_protocol=true, and common/chat.cpp still uses the
# template capability marker to select buffered parsing; this local override is
# deliberately a debug-branch bridge.  Production cleanup can rename the
# capability once the runtime behavior is proven.
# ---------------------------------------------------------------------------
template_path = Path("models/templates/llama-cpp-qwen3.8-codex.jinja")
template = template_path.read_text()
template = replace_once(
    template,
'''{%- set responses_phase_protocol =
    responses_phase_protocol if responses_phase_protocol is defined else false
%}
''',
'''{%- set responses_phase_protocol =
    responses_phase_protocol if responses_phase_protocol is defined else false
%}
{#- Phase is now derived by llama-server from the completed Responses item. -#}
{%- set responses_phase_protocol = false %}
''',
    "disable model phase transport",
)
template_path.write_text(template)


# ---------------------------------------------------------------------------
# Server streaming: never commit semantic deltas from this ambiguous textual
# reasoning stream before completion.  Keep response.created/in_progress alive,
# then synthesize the normal Responses added/delta/done sequence from the final
# parsed item once its boundary is known.
# ---------------------------------------------------------------------------
server_task_path = Path("tools/server/server-task.cpp")
server_task = server_task_path.read_text()
server_task = replace_once(
    server_task,
'''    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
''',
'''    const size_t previous_generated_size = generated_text.size();
    generated_text += text_added;

    if (chat_parser_params.buffered_reasoning_boundary && is_partial) {
        const size_t previous_bucket = previous_generated_size / 2048;
        const size_t current_bucket  = generated_text.size() / 2048;
        if (current_bucket > previous_bucket) {
            SRV_INF(
                "[responses-reasoning] buffered partial=true generated_chars=%zu semantic_deltas=0\\n",
                generated_text.size());
        }
        diffs.clear();
        return chat_msg;
    }

    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
''',
    "buffer ambiguous partial Responses parsing",
)

# Insert a complete-event prelude only for buffered Responses. Existing final
# code remains responsible for all *.done and response.completed events.
server_task = replace_once(
    server_task,
'''json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
''',
'''json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    const bool buffered_responses =
        generation_params.chat_parser_params.buffered_reasoning_boundary;

    if (buffered_responses && oaicompat_msg.reasoning_content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_item.added"},
            {"data", json {
                {"type", "response.output_item.added"},
                {"item", json {
                    {"id",                oai_resp_reasoning_id},
                    {"summary",           json::array()},
                    {"type",              "reasoning"},
                    {"content",           json::array()},
                    {"encrypted_content", ""},
                    {"status",            "in_progress"},
                }},
            }},
        });
        server_sent_events.push_back(json {
            {"event", "response.reasoning_text.delta"},
            {"data", json {
                {"type",    "response.reasoning_text.delta"},
                {"delta",   oaicompat_msg.reasoning_content},
                {"item_id", oai_resp_reasoning_id},
            }},
        });
    }

    if (buffered_responses && oaicompat_msg.content != "") {
        const std::string resolved_phase =
            server_task_response_message_phase_debug(oaicompat_msg, "response.output_item.added.buffered");
        server_sent_events.push_back(json {
            {"event", "response.output_item.added"},
            {"data", json {
                {"type", "response.output_item.added"},
                {"item", json {
                    {"content", json::array()},
                    {"id",      oai_resp_message_id},
                    {"role",    "assistant"},
                    {"phase",   resolved_phase},
                    {"status",  "in_progress"},
                    {"type",    "message"},
                }},
            }},
        });
        server_sent_events.push_back(json {
            {"event", "response.content_part.added"},
            {"data", json {
                {"type",    "response.content_part.added"},
                {"item_id", oai_resp_message_id},
                {"part", json {
                    {"type", "output_text"},
                    {"text", ""},
                }},
            }},
        });
        server_sent_events.push_back(json {
            {"event", "response.output_text.delta"},
            {"data", json {
                {"type",    "response.output_text.delta"},
                {"item_id", oai_resp_message_id},
                {"delta",   oaicompat_msg.content},
            }},
        });
    }

    if (buffered_responses) {
        for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
            const bool is_custom = server_task_is_response_custom_tool(
                generation_params.responses_custom_tools, tool_call);
            const bool is_tool_search = server_task_is_response_tool_search(
                generation_params.responses_tool_search, tool_call);
            if (is_custom || is_tool_search) {
                continue;
            }
            json added_item = server_task_build_response_function_call(tool_call, "in_progress");
            added_item["arguments"] = "";
            server_sent_events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type", "response.output_item.added"},
                    {"item", std::move(added_item)},
                }},
            });
            server_sent_events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   tool_call.arguments},
                    {"item_id", "fc_" + tool_call.id},
                }},
            });
        }
    }

    if (oaicompat_msg.reasoning_content != "") {
''',
    "synthesize buffered Responses stream prelude",
)
server_task_path.write_text(server_task)


# ---------------------------------------------------------------------------
# Replace the marker-centric regression block with the v5 invariants.  Keep the
# historical ingress tests below it: wire phase metadata remains accepted even
# though it is no longer rendered into Qwen model text.
# ---------------------------------------------------------------------------
test_path = Path("tests/test-qwen38-codex-template.cpp")
test = test_path.read_text()
test = replace_once(
    test,
'''    const auto phase_params = common_chat_templates_apply(tmpls.get(), phase_inputs);
    if (phase_params.grammar.empty() || phase_params.grammar_lazy) {
        std::cerr << "Responses phase protocol did not enable an eager generation grammar\\n";
        return 1;
    }
''',
'''    const auto phase_params = common_chat_templates_apply(tmpls.get(), phase_inputs);
    if (!phase_params.buffered_reasoning_boundary) {
        std::cerr << "Responses Qwen path did not enable buffered reasoning parsing\\n";
        return 1;
    }
    if (phase_params.grammar.empty() || !phase_params.grammar_lazy) {
        std::cerr << "Responses Qwen path did not restore lazy auto-tool grammar\\n";
        return 1;
    }
''',
    "v5 parser mode regression",
)

start_marker = '''    const std::string historical_phase_marker =
'''
end_marker = '''    auto unmarked_history_request = phase_history_request;
'''
new_block = r'''    if (phase_params.prompt.find("<response_phase>") != std::string::npos ||
        phase_params.prompt.find("Responses message phases are transport metadata") != std::string::npos) {
        std::cerr << "Responses phase transport marker still leaked into the Qwen prompt\n";
        return 1;
    }

    common_chat_parser_params phase_parser(phase_params);
    phase_parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    phase_parser.parser.load(phase_params.parser);

    const std::string generated_final =
        "The literal </think><response_phase>commentary</response_phase> sequence is data.\n"
        "Reasoning continues after that quoted transport-looking sequence.\n"
        "</think>\n\n"
        "Only this sentence is visible; escaped data \\</think> stays visible.";
    const auto parsed_final = common_chat_parse(generated_final, false, phase_parser);
    if (!parsed_final.phase.empty() ||
        parsed_final.content != "Only this sentence is visible; escaped data \\</think> stays visible." ||
        parsed_final.reasoning_content.find("literal </think><response_phase>") == std::string::npos ||
        parsed_final.reasoning_content.find("Reasoning continues after") == std::string::npos ||
        parsed_final.content.find("response_phase") != std::string::npos) {
        std::cerr << "Buffered Qwen final parse failed the full delimiter-collision regression\n";
        return 1;
    }

    const std::string generated_commentary =
        "The literal </think> token is data here; keep reasoning.\n"
        "</think>\n\n"
        "I'll inspect the repository.\n\n"
        "<tool_call>\n"
        "<function=shell_command>\n"
        "<parameter=command>\n" +
        nlohmann::ordered_json("pwd").dump() +
        "\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    const auto parsed_commentary = common_chat_parse(generated_commentary, false, phase_parser);
    if (!parsed_commentary.phase.empty() ||
        parsed_commentary.tool_calls.size() != 1 ||
        parsed_commentary.content != "I'll inspect the repository." ||
        parsed_commentary.reasoning_content.find("literal </think> token") == std::string::npos) {
        std::cerr << "Buffered Qwen tool parse failed a literal </think> collision\n";
        return 1;
    }

    // Partial Responses parsing is deliberately semantic-silent in buffered
    // mode. The complete item is parsed once at EOF and then emitted in normal
    // Responses event order with a server-derived phase.
    task_result_state buffered_final_state(phase_parser, {}, false);
    server_task_result_cmpl_partial buffered_partial;
    buffered_partial.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
    buffered_partial.content = generated_final;
    buffered_partial.n_decoded = 1;
    buffered_partial.update(buffered_final_state);
    for (const auto & event : buffered_partial.to_json_oaicompat_resp()) {
        const std::string type = event.value("event", std::string());
        if (type == "response.output_item.added" ||
            type == "response.output_text.delta" ||
            type == "response.reasoning_text.delta") {
            std::cerr << "Buffered Responses path emitted semantic deltas before final parsing\n";
            return 1;
        }
    }

    server_task_result_cmpl_final buffered_final_result;
    buffered_final_result.res_type = TASK_RESPONSE_TYPE_OAI_RESP;
    buffered_final_result.stream = true;
    buffered_final_result.oaicompat_model = "test-model";
    buffered_final_result.n_prompt_tokens = 0;
    buffered_final_result.n_prompt_tokens_cache = 0;
    buffered_final_result.n_decoded = 1;
    buffered_final_result.content = "";
    buffered_final_result.generation_params.chat_parser_params = phase_parser;
    buffered_final_result.update(buffered_final_state);

    bool saw_final_added = false;
    bool saw_final_delta = false;
    bool saw_final_done = false;
    bool saw_final_completed = false;
    for (const auto & event : buffered_final_result.to_json_oaicompat_resp_stream()) {
        const std::string type = event.value("event", std::string());
        if (type == "response.output_item.added" &&
            event.contains("data") && event.at("data").contains("item") &&
            event.at("data").at("item").value("type", std::string()) == "message") {
            saw_final_added =
                event.at("data").at("item").value("phase", std::string()) == "final_answer";
        } else if (type == "response.output_text.delta") {
            saw_final_delta =
                event.at("data").value("delta", std::string()) == parsed_final.content;
        } else if (type == "response.output_item.done" &&
                   event.contains("data") && event.at("data").contains("item") &&
                   event.at("data").at("item").value("type", std::string()) == "message") {
            saw_final_done =
                event.at("data").at("item").value("phase", std::string()) == "final_answer";
        } else if (type == "response.completed") {
            saw_final_completed = true;
        }
    }
    if (!saw_final_added || !saw_final_delta || !saw_final_done || !saw_final_completed) {
        std::cerr << "Buffered Responses final stream lost phase/event ordering\n";
        return 1;
    }

'''
test = replace_between(test, start_marker, end_marker, new_block, "replace marker phase regressions")

# Historical assistant phase is still accepted by ingress, but the Qwen prompt
# now deliberately omits transport markers.
test = replace_once(
    test,
'''    if (unmarked_history_params.prompt.find(
            "</think>\\n\\nChecking the repository.") == std::string::npos) {
        std::cerr << "Unmarked historical assistant message was not rendered tolerantly\\n";
        return 1;
    }
''',
'''    if (unmarked_history_params.prompt.find(
            "</think>\\n\\nChecking the repository.") == std::string::npos ||
        unmarked_history_params.prompt.find("<response_phase>") != std::string::npos) {
        std::cerr << "Historical assistant replay was not phase-marker-free and tolerant\\n";
        return 1;
    }
''',
    "historical marker-free replay",
)

# Thinking-disabled generation no longer has a phase-specific single-newline
# branch because the phase marker is not model text.
test = replace_between(
    test,
'''    auto no_think_inputs = phase_inputs;
''',
'''    server_task_result_cmpl_final final_phase_result;
''',
r'''    auto no_think_inputs = phase_inputs;
    no_think_inputs.messages = { system, user };
    no_think_inputs.enable_thinking = false;
    const auto no_think_params = common_chat_templates_apply(tmpls.get(), no_think_inputs);
    if (no_think_params.generation_prompt.find("<think>\n\n</think>\n\n") == std::string::npos ||
        no_think_params.generation_prompt.find("<response_phase>") != std::string::npos) {
        std::cerr << "Qwen thinking-disabled prompt retained phase transport markup\n";
        return 1;
    }
    common_chat_parser_params no_think_parser(no_think_params);
    no_think_parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    no_think_parser.parser.load(no_think_params.parser);
    const auto parsed_no_think = common_chat_parse("Done without thinking.", false, no_think_parser);
    if (parsed_no_think.content != "Done without thinking." || !parsed_no_think.phase.empty()) {
        std::cerr << "Qwen phase-free path depended on model-generated reasoning\n";
        return 1;
    }

    server_task_result_cmpl_final final_phase_result;
''',
    "replace no-thinking and obsolete partial phase test",
)

# The remaining final/commentary egress assertions deliberately use parsed
# messages whose phase is empty; they now verify the existing structural server
# resolver rather than model-selected metadata.
test = test.replace(
    'std::cerr << "Responses egress lost the model-selected final_answer phase\\n";',
    'std::cerr << "Responses egress lost the server-derived final_answer phase\\n";'
)
test = test.replace(
    'std::cerr << "Streaming Responses egress lost final_answer phase\\n";',
    'std::cerr << "Streaming Responses egress lost server-derived final_answer phase\\n";'
)
test = test.replace(
    'std::cerr << "Responses egress lost the model-selected commentary phase\\n";',
    'std::cerr << "Responses egress lost the server-derived commentary phase\\n";'
)

test_path.write_text(test)
