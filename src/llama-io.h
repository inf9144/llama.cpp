#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

static constexpr uint32_t LLAMA_STATE_SEQ_IO_MAGIC = 0xaf143cd8;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);

    // Format errors survive cleanup code that catches and replaces exceptions.
    bool fail_invalid_data() { invalid_data = true; return false; }
    bool has_invalid_data() const { return invalid_data; }

private:
    bool invalid_data = false;
};
