from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one marker in {path}, found {count}")
    p.write_text(text.replace(old, new, 1))


# Opt the dedicated Qwen3.8 Codex template into the private phase transport.
replace_once(
    "models/templates/llama-cpp-qwen3.8-codex.jinja",
    '{#- llama.cpp:xml-string-args=json -#}\n',
    '{#- llama.cpp:xml-string-args=json -#}\n{#- llama.cpp:responses-phase=marker-v1 -#}\n',
)

# The Qwen3-Coder parser is shared by Qwen3.5 and other templates. Responses
# requests alone must not enable this protocol; require explicit template opt-in.
replace_once(
    "common/chat.cpp",
    '''    const bool responses_phase_protocol = inputs.extra_context.is_object() &&
        inputs.extra_context.value("responses_phase_protocol", false);
''',
    '''    const bool responses_phase_protocol =
        tmpl.source().find("llama.cpp:responses-phase=marker-v1") != std::string::npos &&
        inputs.extra_context.is_object() &&
        inputs.extra_context.value("responses_phase_protocol", false);
''',
)

# Focused gate: prove the selected Codex template advertises the capability.
replace_once(
    "tests/test-qwen38-codex-template.cpp",
    '''    const auto caps = common_chat_templates_get_caps(tmpls.get());
''',
    '''    const std::string template_source = read_file(template_path);
    if (template_source.find("llama.cpp:responses-phase=marker-v1") == std::string::npos) {
        std::cerr << "Codex template did not opt into the Responses phase protocol\\n";
        return 1;
    }

    const auto caps = common_chat_templates_get_caps(tmpls.get());
''',
)
