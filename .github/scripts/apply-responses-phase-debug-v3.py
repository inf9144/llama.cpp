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


# ---------------------------------------------------------------------------
# Split tolerant runtime parsing from strict generation grammar.
# ---------------------------------------------------------------------------
chat_path = Path("common/chat.cpp")
chat = chat_path.read_text()

chat = replace_in_qwen(
    chat,
'''    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
''',
'''    auto build_parser = [&](bool strict_phase_generation) {
        return build_chat_peg_parser([&, strict_phase_generation](common_chat_peg_builder & p) {
''',
    "start split Qwen parser builder",
)

chat = replace_in_qwen(
    chat,
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
'''        auto phase = p.eps();
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
                // Runtime parser hardening only: a literal </think> mentioned in
                // reasoning is not a boundary unless a phase opener follows it.
                // This protects streaming from leaking the remaining reasoning.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until("</think>\\n<response_phase>")) +
                                       p.literal("</think>") + p.space());
            } else {
                // Generation grammar keeps Qwen's native reasoning boundary.
                // The required phase parser follows this close independently.
                reasoning = p.optional("<think>" + p.space() +
                                       p.reasoning(p.until_one_of({ "</think>", "<tool_call>" })) +
                                       (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));
            }
        }
''',
    "separate runtime reasoning boundary from generation phase",
)

chat = replace_in_qwen(
    chat,
'''        // Content only parser
        return generation_prompt + (reasoning << phase << p.content(p.rest()));
    });

    data.parser = parser.save();
''',
'''        // Content only parser
        return generation_prompt + (reasoning << phase << p.content(p.rest()));
        });
    };

    // Streamed output uses the collision-hardened tolerant parser. Fresh
    // Responses generation gets a separate strict grammar below.
    auto parser = build_parser(false);
    data.parser = parser.save();
''',
    "finish split Qwen parser builder",
)

chat = replace_in_qwen(
    chat,
'''        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });
''',
'''        auto grammar_parser = build_parser(responses_phase_protocol);
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            grammar_parser.build_grammar(builder, data.grammar_lazy);
        });
''',
    "use strict parser for Responses generation grammar",
)

chat_path.write_text(chat)


# ---------------------------------------------------------------------------
# Runtime reasoning diagnostics.
# ---------------------------------------------------------------------------
server_task_path = Path("tools/server/server-task.cpp")
server_task = server_task_path.read_text()
server_task = replace_once(
    server_task,
'''    if (!new_msg.empty()) {
        if (new_msg.phase != msg_prv_copy.phase) {
''',
'''    if (!new_msg.empty()) {
        const size_t previous_reasoning_bucket = msg_prv_copy.reasoning_content.size() / 2048;
        const size_t current_reasoning_bucket  = new_msg.reasoning_content.size() / 2048;
        if (current_reasoning_bucket > previous_reasoning_bucket) {
            SRV_INF(
                "[responses-reasoning] parser partial=%s reasoning_chars=%zu generated_chars=%zu phase=%s content_chars=%zu tool_calls=%zu\\n",
                is_partial ? "true" : "false",
                new_msg.reasoning_content.size(),
                generated_text.size(),
                new_msg.phase.empty() ? "<none>" : new_msg.phase.c_str(),
                new_msg.content.size(),
                new_msg.tool_calls.size());
        }
        if (new_msg.phase != msg_prv_copy.phase) {
''',
    "reasoning progress diagnostics",
)
server_task_path.write_text(server_task)


# Log the request effort, the effort instruction actually rendered into the
# model prompt, and the independent reasoning token budget.
server_common_path = Path("tools/server/server-common.cpp")
server_common = server_common_path.read_text()
server_common = replace_once(
    server_common,
'''    // Reasoning budget: pass parameters through to sampling layer
    {
        int reasoning_budget = json_value(body, "reasoning_budget_tokens",
                               json_value(body, "thinking_budget_tokens", -1));
        if (reasoning_budget == -1) {
            reasoning_budget = opt.reasoning_budget;
        }

        if (!chat_params.thinking_end_tags.empty()) {
            llama_params["reasoning_budget_tokens"] = reasoning_budget;
            llama_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
            llama_params["reasoning_budget_end_tags"] = chat_params.thinking_end_tags;
            llama_params["reasoning_budget_message"] = json_value(body, "reasoning_budget_message", opt.reasoning_budget_message);
            llama_params["reasoning_control"] = json_value(body, "reasoning_control", false);
        }
    }
''',
'''    // Reasoning budget: pass parameters through to sampling layer.
    // reasoning_effort is a template instruction; this is an independent
    // token budget and normally remains -1 (unlimited).
    {
        int reasoning_budget = json_value(body, "reasoning_budget_tokens",
                               json_value(body, "thinking_budget_tokens", -1));
        if (reasoning_budget == -1) {
            reasoning_budget = opt.reasoning_budget;
        }

        if (!chat_params.thinking_end_tags.empty()) {
            llama_params["reasoning_budget_tokens"] = reasoning_budget;
            llama_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
            llama_params["reasoning_budget_end_tags"] = chat_params.thinking_end_tags;
            llama_params["reasoning_budget_message"] = json_value(body, "reasoning_budget_message", opt.reasoning_budget_message);
            llama_params["reasoning_control"] = json_value(body, "reasoning_control", false);
        }

        const std::string request_effort = json_value(body, "reasoning_effort", std::string());
        std::string rendered_effort = "<none>";
        if (chat_params.prompt.find("Reasoning effort is set to xhigh.") != std::string::npos) {
            rendered_effort = "xhigh";
        } else if (chat_params.prompt.find("Reasoning effort is set to medium.") != std::string::npos) {
            rendered_effort = "medium";
        } else if (chat_params.prompt.find("Reasoning effort is set to low.") != std::string::npos) {
            rendered_effort = "low";
        }
        SRV_INF(
            "[responses-reasoning] request_effort=%s rendered_effort=%s enable_thinking=%s reasoning_format=%s budget_tokens=%d thinking_end_tags=%zu\\n",
            request_effort.empty() ? "<unset>" : request_effort.c_str(),
            rendered_effort.c_str(),
            inputs.enable_thinking ? "true" : "false",
            common_reasoning_format_name(inputs.reasoning_format),
            reasoning_budget,
            chat_params.thinking_end_tags.size());
    }
''',
    "request reasoning diagnostics",
)
server_common_path.write_text(server_common)


# ---------------------------------------------------------------------------
# Regression coverage for split generation/runtime parsing and effort bridge.
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

    const std::string historical_phase_marker =
''',
'''    const auto phase_params = common_chat_templates_apply(tmpls.get(), phase_inputs);
    if (phase_params.grammar.empty() || phase_params.grammar_lazy) {
        std::cerr << "Responses phase protocol did not enable an eager generation grammar\\n";
        return 1;
    }

    // Codex Responses reasoning.effort must survive wire conversion and reach
    // the actual Qwen model prompt.
    auto medium_reasoning_request = phase_history_request;
    medium_reasoning_request["reasoning"] = {{"effort", "medium"}};
    const auto converted_medium_reasoning =
        server_chat_convert_responses_to_chatcmpl(medium_reasoning_request);
    if (converted_medium_reasoning.value("reasoning_effort", std::string()) != "medium") {
        std::cerr << "Responses reasoning.effort did not map to reasoning_effort\\n";
        return 1;
    }
    auto medium_reasoning_inputs = phase_inputs;
    medium_reasoning_inputs.chat_template_kwargs["reasoning_effort"] =
        nlohmann::ordered_json("medium").dump();
    const auto medium_reasoning_params =
        common_chat_templates_apply(tmpls.get(), medium_reasoning_inputs);
    if (medium_reasoning_params.prompt.find("Reasoning effort is set to medium.") == std::string::npos ||
        medium_reasoning_params.prompt.find("avoid exploring low-value alternatives") == std::string::npos) {
        std::cerr << "Qwen prompt did not receive the medium reasoning instruction\\n";
        return 1;
    }

    const std::string historical_phase_marker =
''',
    "reasoning effort bridge regression",
)

test = replace_once(
    test,
'''    // Current Responses generations are strict: an unmarked generated message
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
''',
'''    // Runtime parsing remains tolerant/collision-safe. Fresh generation is
    // strict through phase_params.grammar; malformed unmarked output must not
    // leak visible content from the runtime parser.
    bool unmarked_content_hidden = false;
    try {
        const auto parsed_unmarked_generation = common_chat_parse(
            "Legacy reasoning.\\n</think>\\n\\nLegacy final text.", false, phase_parser);
        unmarked_content_hidden =
            parsed_unmarked_generation.content.empty() &&
            parsed_unmarked_generation.tool_calls.empty();
    } catch (const std::exception &) {
        unmarked_content_hidden = true;
    }
    if (!unmarked_content_hidden) {
        std::cerr << "Responses runtime parser exposed unmarked generated content\\n";
        return 1;
    }
''',
    "split parser strictness regression",
)

test_path.write_text(test)
