from pathlib import Path

path = Path("common/chat.cpp")
text = path.read_text()
old = '''    const bool responses_phase_protocol = inputs.extra_context.has_value() &&
        inputs.extra_context->value("responses_phase_protocol", false);
'''
new = '''    const bool responses_phase_protocol = inputs.extra_context.is_object() &&
        inputs.extra_context.value("responses_phase_protocol", false);
'''
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected exactly one Responses phase context marker, found {count}")
path.write_text(text.replace(old, new, 1))
