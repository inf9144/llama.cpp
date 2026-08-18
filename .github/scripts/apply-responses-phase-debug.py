from pathlib import Path

path = Path("tools/server/server-task.cpp")
text = path.read_text()


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
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

replace_once(
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

replace_once(
'''            {"phase",  server_task_response_message_phase(msg)},
''',
'''            {"phase",  server_task_response_message_phase_debug(msg, "response.completed.output")},
''',
"non-stream phase",
)

replace_once(
'''            {"phase",   server_task_response_message_phase(oaicompat_msg)}
''',
'''            {"phase",   server_task_response_message_phase_debug(oaicompat_msg, "response.output_item.done")}
''',
"stream done phase",
)

replace_once(
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

path.write_text(text)
