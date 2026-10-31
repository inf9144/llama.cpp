# GGML Operations

List of GGML operations and backend support status.

## How to add a backend to this table:

1. Run `test-backend-ops support --output csv` with your backend name and redirect output to a csv file in `docs/ops/` (e.g., `docs/ops/CUDA.csv`)
2. Regenerate `/docs/ops.md` via `./scripts/create_ops_docs.py`

Legend:
- ✅ Fully supported by this backend
- 🟡 Partially supported by this backend
- ❌ Not supported by this backend

| Operation | BLAS | CANN | CPU | CUDA | ET | HTP | MTL | OpenCL | SYCL | Vulkan | WebGPU | ZenDNN | zDNN |
|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|                              ABS | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              ACC | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              ADD | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             ADD1 | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                           ADD_ID | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           ARANGE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           ARGMAX | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          ARGSORT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             CEIL | ❌ | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            CLAMP | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        COL2IM_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CONCAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                             CONT | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                       CONV_1D_DW | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
|                          CONV_2D | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                       CONV_2D_DW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          CONV_3D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_2D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              COS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      COUNT_EQUAL | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              CPY | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|               CROSS_ENTROPY_LOSS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|          CROSS_ENTROPY_LOSS_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CUMSUM | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             DIAG | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    DIAG_MASK_INF | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              DIV | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     DSV4_HC_COMB | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     DSV4_HC_POST | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                      DSV4_HC_PRE | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              DUP | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              ELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              EXP | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            EXPM1 | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             FILL | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                   FLASH_ATTN_EXT | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                            FLOOR | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                  GATED_DELTA_NET | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                GATED_LINEAR_ATTN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            GEGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        GEGLU_ERF | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      GEGLU_QUICK | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             GELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GELU_ERF | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                       GELU_QUICK | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GET_ROWS | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ❌ | ❌ |
|                    GET_ROWS_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |
|                       GROUP_NORM | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                      HARDSIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        HARDSWISH | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           IM2COL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        IM2COL_3D | ❌ | ❌ | ✅ | ✅ | ❌ | 🟡 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          L2_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                       LEAKY_RELU | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                LIGHTNING_INDEXER | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              LOG | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             MEAN | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                              MUL | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          MUL_MAT | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
|                 MUL_MAT_HADAMARD | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 |
|                       MUL_MAT_ID | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 | ❌ |
|                  MUL_MAT_ID_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                  MUL_MAT_ID_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              NEG | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                   OPT_STEP_ADAMW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                     OPT_STEP_SGD | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         OUT_PROD | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |
|                              PAD | ❌ | 🟡 | ✅ | 🟡 | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                   PAD_REFLECT_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                          POOL_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          POOL_2D | ❌ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            REGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             RELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           REPEAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                      REPEAT_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         RMS_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    RMS_NORM_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                             ROLL | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                             ROPE | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        ROPE_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            ROUND | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        RWKV_WKV6 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                        RWKV_WKV7 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            SCALE | ❌ | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SET | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                         SET_ROWS | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                              SGN | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          SIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SILU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        SILU_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                              SIN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SOFTPLUS | ❌ | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SOFT_MAX | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    SOFT_MAX_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                        SOLVE_TRI | ❌ | ❌ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                              SQR | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SQRT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_CONV | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_SCAN | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ❌ | ❌ |
|                             STEP | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUB | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUM | ❌ | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                         SUM_ROWS | ❌ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                           SWIGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     SWIGLU_CLAMP | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                       SWIGLU_OAI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             TANH | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|               TIMESTEP_EMBEDDING | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            TOP_K | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                              TRI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            TRUNC | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          UPSCALE | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            XIELU | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |

## Native QSA in this fork

`QSA_SELECT` forms a compact pool/tail union without a dense attention mask. `FLASH_ATTN_SPARSE` consumes per-query cell indices. The CPU backend provides reference implementations; HIP implements Q8_0 K/V, 256-element heads and GQA ratios divisible by four on gfx1100/gfx1101. CUDA, Vulkan and other backends retain the existing Qwen4Exp mask path. HIP execution and performance validation remain pending on physical hardware; no HIP support CSV is claimed from CPU results.

See [the implementation and validation report](development/hip-qsa-sparse-attention.md).

The custom CUDA/HIP radix `TOP_K` path resolves identical cutoff keys by lower input-column index. Its output order remains unspecified; the bitonic, NVIDIA CUB and other backend paths retain their existing tie handling. This stabilizes QSA pool membership without changing unequal-score ranking. See [the physical replay result and cutoff regression](development/moe-cache-prefill.md#physical-topk-result-and-deterministic-cutoff-selection). No new support CSV result is claimed without hardware validation.

## Internal MoE operations in this fork

`MUL_MAT_ID_STREAM` uses bounded expert-address packets on supported RDNA3 HIP configurations; CPU has a test oracle. `MOE_CACHE_LOOKUP` resolves resident slots and publishes asynchronous demand/miss notifications on RDNA3 HIP with stream-signal graph support. CPU, CUDA, Vulkan and RPC do not execute the lookup operation; model construction retains their existing paths. Both operations contain process-local state and are rejected by RPC. These are implementation capabilities, not generated hardware support CSV results.

See [the MoE implementation and validation report](development/moe-cache-prefill.md).
