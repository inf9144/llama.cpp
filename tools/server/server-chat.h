// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include <nlohmann/json_fwd.hpp>

using json = nlohmann::ordered_json;

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
