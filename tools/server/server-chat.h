// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include "json.h"

inline uint64_t server_chat_responses_fingerprint(const std::string & value) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : value) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline const char * server_chat_json_type_name(const json & value) {
    if (value.is_null()) {
        return "null";
    }
    if (value.is_object()) {
        return "object";
    }
    if (value.is_array()) {
        return "array";
    }
    if (value.is_string()) {
        return "string";
    }
    if (value.is_boolean()) {
        return "boolean";
    }
    if (value.is_number()) {
        return "number";
    }
    return "unknown";
}

inline void server_chat_responses_content_stats(
        const json & content,
        size_t & parts,
        size_t & text_chars,
        size_t & images,
        size_t & other) {
    if (content.is_string()) {
        parts = 1;
        text_chars = content.get<std::string>().size();
        return;
    }
    if (!content.is_array()) {
        if (!content.is_null()) {
            other = 1;
        }
        return;
    }

    parts = content.size();
    for (const json & part : content) {
        if (!part.is_object()) {
            ++other;
            continue;
        }
        if (part.contains("text") && part.at("text").is_string()) {
            text_chars += part.at("text").get<std::string>().size();
            continue;
        }
        if ((part.contains("image_url") && !part.at("image_url").is_null()) ||
            json_value(part, "type", std::string()) == "input_image") {
            ++images;
            continue;
        }
        ++other;
    }
}

inline void server_chat_log_responses_structure(const json & body) {
    const json & input = body.at("input");
    bool is_compaction = false;
    if (input.is_array()) {
        for (const json & item : input) {
            if (item.is_object() && json_value(item, "type", std::string()) == "compaction_trigger") {
                is_compaction = true;
                break;
            }
        }
    }

    std::string instructions;
    if (body.contains("instructions") && body.at("instructions").is_string()) {
        instructions = body.at("instructions").get<std::string>();
    }

    std::string tools_serialized;
    size_t tools_count = 0;
    if (body.contains("tools")) {
        tools_serialized = body.at("tools").dump();
        if (body.at("tools").is_array()) {
            tools_count = body.at("tools").size();
        }
    }

    const char * input_kind = server_chat_json_type_name(input);
    const size_t input_count = input.is_array() ? input.size() : input.is_null() ? 0 : 1;
    SRV_INF(
        "Responses structure: kind=%s instructions={chars=%zu,hash=%016llx} tools={count=%zu,bytes=%zu,hash=%016llx} input={kind=%s,count=%zu}\n",
        is_compaction ? "compaction" : "normal",
        instructions.size(),
        (unsigned long long) server_chat_responses_fingerprint(instructions),
        tools_count,
        tools_serialized.size(),
        (unsigned long long) server_chat_responses_fingerprint(tools_serialized),
        input_kind,
        input_count);

    if (input.is_string()) {
        SRV_INF("Responses input[0]: type=input_text role=user text_chars=%zu\n", input.get<std::string>().size());
        return;
    }
    if (!input.is_array()) {
        return;
    }

    for (size_t i = 0; i < input.size(); ++i) {
        const json & item = input.at(i);
        if (!item.is_object()) {
            SRV_INF("Responses input[%zu]: type=%s\n", i, server_chat_json_type_name(item));
            continue;
        }

        std::string type = json_value(item, "type", std::string());
        const std::string role = json_value(item, "role", std::string());
        const std::string name = json_value(item, "name", std::string());
        if (type.empty() && !role.empty()) {
            type = "message";
        } else if (type.empty()) {
            type = "object";
        }

        size_t content_parts = 0;
        size_t text_chars = 0;
        size_t images = 0;
        size_t other_parts = 0;
        if (item.contains("content")) {
            server_chat_responses_content_stats(
                item.at("content"), content_parts, text_chars, images, other_parts);
        }

        size_t arguments_chars = 0;
        if (item.contains("arguments") && item.at("arguments").is_string()) {
            arguments_chars = item.at("arguments").get<std::string>().size();
        }

        size_t output_parts = 0;
        size_t output_chars = 0;
        size_t output_images = 0;
        size_t output_other = 0;
        if (item.contains("output")) {
            server_chat_responses_content_stats(
                item.at("output"), output_parts, output_chars, output_images, output_other);
        }

        size_t encrypted_chars = 0;
        if (item.contains("encrypted_content") && item.at("encrypted_content").is_string()) {
            encrypted_chars = item.at("encrypted_content").get<std::string>().size();
        }

        SRV_INF(
            "Responses input[%zu]: type=%s role=%s name=%s content={parts=%zu,text_chars=%zu,images=%zu,other=%zu} arguments_chars=%zu output={parts=%zu,text_chars=%zu,images=%zu,other=%zu} encrypted_chars=%zu\n",
            i,
            type.c_str(),
            role.empty() ? "-" : role.c_str(),
            name.empty() ? "-" : name.c_str(),
            content_parts,
            text_chars,
            images,
            other_parts,
            arguments_chars,
            output_parts,
            output_chars,
            output_images,
            output_other,
            encrypted_chars);
    }
}

// Intercept the string-valued previous_response_id lookup at the start of the
// Responses converter. This keeps diagnostics on the raw wire body without
// changing conversion or cache behavior.
inline std::string json_value(const json & body, const std::string & key, const std::string & default_value) {
    if (key == "previous_response_id" && body.contains("input")) {
        server_chat_log_responses_structure(body);
    }
    return ::json_value<std::string>(body, key, default_value);
}

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json server_chat_convert_responses_to_chatcmpl(const json & body);

// Encode/decode namespaced Responses tools to a flat Chat Completions function name.
std::string server_chat_encode_namespace_tool_name(const std::string & tool_namespace, const std::string & tool_name);
bool server_chat_decode_namespace_tool_name(const std::string & flat_name, std::string & tool_namespace, std::string & tool_name);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json server_chat_convert_anthropic_to_oai(const json & body);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
