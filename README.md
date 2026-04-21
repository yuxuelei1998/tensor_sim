# TensorSim

TensorSim is a **bit-accurate C++ simulator** for NVIDIA GPU Tensor Cores. It reproduces the bit-exact behavior of each GPU generation's mixed-precision dot-product-accumulate operations according to hardware specifications, including internal representation precision, alignment-truncation policy, rounding modes, and special-value handling. It is intended for algorithm validation, precision research, and cross-architecture comparison.

---

## Table of Contents

- [Implemented Architectures](#implemented-architectures)
- [Floating-Point Format Reference](#floating-point-format-reference)
- [Operation Semantics](#operation-semantics)
  - [Common Conventions](#common-conventions)
  - [Volta — DP4A](#volta--dp4a)
  - [Ampere — DP8A](#ampere--dp8a)
  - [Hopper — DP16A / DP32A](#hopper--dp16a--dp32a)
  - [Blackwell — DP16A / DP32A / DP64A](#blackwell--dp16a--dp32a--dp64a)
  - [Custom — Fully Configurable DPnA](#custom--fully-configurable-dpna)
- [Project Structure](#project-structure)
- [Building](#building)
- [Running Tests](#running-tests)
- [Interactive CLI Tool](#interactive-cli-tool)
- [Test Vector File Format](#test-vector-file-format)

---

## Implemented Architectures

| Architecture | Operation | Group Size | A/B Input Type | Accumulator (C/D) | Rounding |
|--------------|-----------|-----------|---------------|-------------------|----------|
| Volta     | DP4A (4-point dot-product accumulate)  |  4 | FP16     | FP32 | TC-Truncation (round toward zero) |
| Volta     | DP4A (4-point dot-product accumulate)  |  4 | FP16     | FP16 | Round-to-nearest-even (RNE) |
| Ampere    | DP8A (8-point dot-product accumulate)  |  8 | FP16     | FP32 | TC-Truncation |
| Ampere    | DP8A (8-point dot-product accumulate)  |  8 | FP16     | FP16 | RNE |
| Ampere    | DP8A (8-point dot-product accumulate)  |  8 | BF16     | FP32 | TC-Truncation |
| Hopper    | DP16A (16-point dot-product accumulate)| 16 | FP16     | FP32 | TC-Truncation |
| Hopper    | DP16A (16-point dot-product accumulate)| 16 | FP16     | FP16 | RNE |
| Hopper    | DP16A (16-point dot-product accumulate)| 16 | BF16     | FP32 | TC-Truncation |
| Hopper    | DP32A (32-point dot-product accumulate)| 32 | FP8 E5M2 | FP32 | TC-Truncation |
| Hopper    | DP32A (32-point dot-product accumulate)| 32 | FP8 E5M2 | FP16 | RNE |
| Hopper    | DP32A (32-point dot-product accumulate)| 32 | FP8 E4M3 | FP32 | TC-Truncation |
| Hopper    | DP32A (32-point dot-product accumulate)| 32 | FP8 E4M3 | FP16 | RNE |
| Blackwell | DP16A (16-point dot-product accumulate) |  16 | FP16          | FP32 | TC-Truncation |
| Blackwell | DP16A (16-point dot-product accumulate) |  16 | FP16          | FP16 | RNE |
| Blackwell | DP16A (16-point dot-product accumulate) |  16 | BF16          | FP32 | TC-Truncation |
| Blackwell | DP32A (32-point dot-product accumulate) |  32 | FP8 E5M2      | FP32 | TC-Truncation |
| Blackwell | DP32A (32-point dot-product accumulate) |  32 | FP8 E5M2      | FP16 | RNE |
| Blackwell | DP32A (32-point dot-product accumulate) |  32 | FP8 E4M3      | FP32 | TC-Truncation |
| Blackwell | DP32A (32-point dot-product accumulate) |  32 | FP8 E4M3      | FP16 | RNE |
| Blackwell | DP32A (32-point dot-product accumulate) |  32 | FP4 E2M1      | FP32 | TC-Truncation |
| Blackwell | DP32A (32-point dot-product accumulate) |  32 | FP4 E2M1      | FP16 | RNE |
| Blackwell | MXFP4 (DP64A + UE8M0 block scaling)    |  64 | FP4 E2M1 + UE8M0  | FP32 | TC-Truncation |
| Blackwell | NVFP4 (DP64A + UE4M3 block scaling)    |  64 | FP4 E2M1 + UE4M3  | FP32 | TC-Truncation |
| Custom    | DPnA (configurable n-point dot-product) |   n | FP16/BF16/FP8/FP4 | FP32/FP16 | RTZ or RNE |

---

## Floating-Point Format Reference

**Floating-point element formats:**

| Format     | Bits | Sign | Exponent (bias) | Mantissa | Max normal value  | NaN / Inf |
|------------|------|------|-----------------|----------|-------------------|-----------|
| FP32       | 32   | 1    | 8 (bias 127)    | 23       | ≈ 3.40 × 10³⁸    | Yes |
| FP16       | 16   | 1    | 5 (bias 15)     | 10       | 65504             | Yes |
| BF16       | 16   | 1    | 8 (bias 127)    |  7       | ≈ 3.39 × 10³⁸    | Yes |
| FP8 E5M2   |  8   | 1    | 5 (bias 15)     |  2       | 57344             | Yes |
| FP8 E4M3   |  8   | 1    | 4 (bias 7)      |  3       | 448               | NaN only (no Inf) |
| FP4 E2M1   |  4   | 1    | 2 (bias 1)      |  1       | 6.0               | None |

**Scale factor formats** (used in MXFP4 and NVFP4 microscaling modes):

| Format  | Bits | Type | Encoding | Multiplier | Special values |
|---------|------|------|----------|------------|----------------|
| UE8M0   |  8   | Pure unsigned exponent | Integer byte `v` | 2^(v − 127) | `0x00` → zero, `0xFF` → NaN |
| UE4M3   |  7   | Unsigned float (4-bit exp, 3-bit mant) | `{exp[3:0], mant[2:0]}` | Decoded FP value | `0x00` → zero, `0x7F`/`0xFF` → NaN |

> **FP8 E4M3 special convention (NVIDIA)**: only `0x7F` (`+NaN`) and `0xFF` (`−NaN`) encode NaN; there is no Inf encoding; maximum normal value is 448.
>
> **FP4 E2M1**: no NaN or Inf encodings; subnormal representable value is ±0.5; normal range ±[0.5, 6.0].
>
> **UE4M3 decoding**: for byte `v` (7-bit body = `v & 0x7F`), if exponent field is nonzero the value is `(8 | mant) × 2^(exp − 10)`; if exponent field is zero the value is `mant × 2^(−9)` (subnormal).

---

## Operation Semantics

### Common Conventions

The following rules apply to all architectures and all modes:

- **Negative zero treated as positive zero**: `-0.0` is equivalent to `+0.0` in all computations and does not trigger special-value logic.
- **Full subnormal support**: both inputs and outputs may be subnormal; the simulator performs complete normalization.
- **Special-value output**: if any input is NaN, a `0 × ∞` product occurs, or `+∞ + (−∞)` is encountered, a fixed exception sentinel is written directly (NaN payload is not propagated):
  - FP32 accumulator: `0x7FFFFFFF`
  - FP16 accumulator: `0x7FFF`
- **Overflow produces infinity**: if the accumulated result exceeds the output format's representable range, `±Inf` is produced.
- **Arbitrary vector length**: the public API accepts vectors of any length. Elements at the end of an incomplete group are zero-padded, and groups are chained serially (each group's output D becomes the next group's C input).

---

### Volta — DP4A

**Operation**

$$D = \sum_{i=0}^{3} A_i \times B_i + C$$

Each call processes one group of 4 elements (**DP4A**).

**Internal Representation**

Each product $p_k = A_k \times B_k$ and the accumulator C are held in aligned form before summation:

$$(-1)^s \times (1 + m/2^{23}) \times 2^e$$

where the mantissa field $m$ is **23 bits** and $e$ is the unbiased exponent.

**Exponent Alignment and Truncation**

All 5 intermediate values (4 products + C) are aligned to the maximum exponent $e_{\max}$:

- If a value must be right-shifted by $s \geq 24$ bits (the entire 24-bit significand falls outside the window), it is **discarded entirely** and contributes neither to the sum nor to the sticky bit.

**Accumulation and Normalization**

After alignment, the values are summed as integers. The result is normalized in **one step** and then converted to the output format using the mode's rounding rule:

| Mode | Internal exponent range | Rounding |
|------|-------------------------|----------|
| FP32 accumulator | Full FP32 range (8-bit biased exponent) | TC-Truncation |
| FP16 accumulator | FP16 range (5-bit biased exponent); out-of-range clamped to ±Inf/0 | RNE |

**Public API**

```cpp
fp32_t volta_dp4a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);

fp16_t volta_dp4a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c);
```

---

### Ampere — DP8A

**Operation**

$$D = \sum_{i=0}^{7} A_i \times B_i + C$$

Each call processes one group of 8 elements (**DP8A**).

**Internal Representation**

$$(-1)^s \times (1 + m/2^{24}) \times 2^e$$

The mantissa field $m$ is **24 bits** (one more than Volta), giving a 25-bit significand.

**Exponent Alignment and Truncation**

All 9 intermediate values (8 products + C) are aligned to $e_{\max}$:

- If a value must be right-shifted by $s \geq 25$ bits, it is discarded entirely and contributes no sticky bit.

**Variants**

**Variant 1: FP16 inputs + FP32 accumulator**

- Product $p_k = \text{FP16}_k \times \text{FP16}_k$; the product of two 11-bit significands is mapped to the 24-bit mantissa field.
- C is extended to 24 bits by left-shifting the 23-bit FP32 mantissa by 1.
- Rounding: **TC-truncation**.

**Variant 2: FP16 inputs + FP16 accumulator**

- Product significand handling is identical to Variant 1; if the product's unbiased exponent > 15 it is clamped to ±Inf, and if < −14 it enters the subnormal domain.
- C is extended to 24 bits by left-shifting the 10-bit FP16 mantissa by 14.
- Rounding: **RNE**; the sticky bit comes only from the final normalization shift (not from the alignment phase).

**Variant 3: BF16 inputs + FP32 accumulator**

- BF16 has an 8-bit significand (1 + 7 mantissa bits); the product of two 8-bit significands (14–16 bit range) is mapped to the 24-bit mantissa field.
- BF16 and FP32 share the same 8-bit exponent domain (bias 127), so products cannot overflow the FP32 exponent range.
- Rounding: **TC-truncation**.

**Public API**

```cpp
fp32_t ampere_dp8a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);

fp16_t ampere_dp8a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c);

fp32_t ampere_dp8a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c);
```

---

### Hopper — DP16A / DP32A

Hopper introduces two operation widths:

#### DP16A (16-point dot-product accumulate)

**Operation**

$$D = \sum_{i=0}^{15} A_i \times B_i + C$$

**Internal Representation**

The mantissa field $m$ is **25 bits** (one more than Ampere), giving a 26-bit significand.

**Exponent Alignment and Truncation**

All 17 intermediate values (16 products + C) are aligned to $e_{\max}$:

- If a value must be right-shifted by $s \geq 26$ bits, it is discarded entirely.

**Supported Variants**

| Input type | Accumulator | Rounding |
|------------|-------------|----------|
| FP16 | FP32 | TC-Truncation |
| FP16 | FP16 | RNE |
| BF16 | FP32 | TC-Truncation |

**Public API**

```cpp
fp32_t hopper_dp16a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);
fp16_t hopper_dp16a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c);
fp32_t hopper_dp16a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c);
```

#### DP32A (32-point dot-product accumulate, FP8 inputs)

**Operation**

$$D = \sum_{i=0}^{31} A_i \times B_i + C$$

**Internal Representation**

The mantissa field $m$ is **13 bits**, giving a 14-bit significand (a precision-convergence design tailored for FP8 products).

**Exponent Alignment and Truncation**

All 33 intermediate values (32 products + C) are aligned to $e_{\max}$:

- If a value must be right-shifted by $s \geq 14$ bits, it is discarded entirely.

**Supported Variants**

| Input type | Accumulator | Rounding |
|------------|-------------|----------|
| FP8 E5M2 | FP32 | TC-Truncation |
| FP8 E5M2 | FP16 | RNE |
| FP8 E4M3 | FP32 | TC-Truncation |
| FP8 E4M3 | FP16 | RNE |

**Public API**

```cpp
fp32_t hopper_dp32a_e5m2_fp32(const e5m2_t* a, const e5m2_t* b, size_t len, fp32_t c);
fp16_t hopper_dp32a_e5m2_fp16(const e5m2_t* a, const e5m2_t* b, size_t len, fp16_t c);
fp32_t hopper_dp32a_e4m3_fp32(const e4m3_t* a, const e4m3_t* b, size_t len, fp32_t c);
fp16_t hopper_dp32a_e4m3_fp16(const e4m3_t* a, const e4m3_t* b, size_t len, fp16_t c);
```

---

### Blackwell — DP16A / DP32A /DP64A

Blackwell's DP16A is identical to Hopper's (same group size, same 25-bit mantissa). The key upgrade is in DP32A: the internal mantissa precision is raised from Hopper's **13 bits** to **25 bits**, significantly improving FP8 dot-product accuracy. Blackwell also extends **DP32A** to FP4 E2M1 inputs (plain, no scaling), and adds two microscaling variants — **MXFP4** and **NVFP4** — that apply per-block scale factors to FP4 operands using a 64-element group.

#### DP16A (16-point dot-product accumulate)

Same specification as Hopper DP16A (25-bit mantissa, TC-truncation / RNE).

**Public API**

```cpp
fp32_t blackwell_dp16a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp16a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c);
fp32_t blackwell_dp16a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c);
```

#### DP32A (32-point dot-product accumulate, FP8 inputs)

**Internal Representation**

The mantissa field $m$ is upgraded to **25 bits** (12 bits wider than Hopper's DP32A), giving a 26-bit significand.

**Exponent Alignment and Truncation**

- If a value must be right-shifted by $s \geq 26$ bits, it is discarded entirely.

**Supported Variants**

| Input type | Accumulator | Rounding |
|------------|-------------|----------|
| FP8 E5M2 | FP32 | TC-Truncation |
| FP8 E5M2 | FP16 | RNE |
| FP8 E4M3 | FP32 | TC-Truncation |
| FP8 E4M3 | FP16 | RNE |

**Public API**

```cpp
fp32_t blackwell_dp32a_e5m2_fp32(const e5m2_t* a, const e5m2_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp32a_e5m2_fp16(const e5m2_t* a, const e5m2_t* b, size_t len, fp16_t c);
fp32_t blackwell_dp32a_e4m3_fp32(const e4m3_t* a, const e4m3_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp32a_e4m3_fp16(const e4m3_t* a, const e4m3_t* b, size_t len, fp16_t c);
```

#### DP32A (32-point dot-product accumulate, FP4 inputs)

**Operation**

$$D = \sum_{i=0}^{31} A_i \times B_i + C$$

**Internal Representation**

The mantissa field $m$ is **25 bits** (26-bit significand), shared with DP16A and FP8 DP32A. FP4 E2M1 has a 2-bit significand (1 implicit + 1 mantissa bit).

**Exponent Alignment and Truncation**

All 33 intermediate values (32 products + C) are aligned to $e_{\max}$:

- If a value must be right-shifted by $s \geq 26$ bits, it is discarded entirely.

**FP4 E2M1 special value handling**

FP4 E2M1 has no NaN or Inf encodings. The only special value is zero (all mantissa bits zero). The subnormal value (exponent = 0, mantissa = 1) represents ±0.5.

**Supported Variants**

| Input type | Accumulator | Rounding |
|------------|-------------|----------|
| FP4 E2M1 | FP32 | TC-Truncation |
| FP4 E2M1 | FP16 | RNE |

**Public API**

```cpp
fp32_t blackwell_dp32a_e2m1_fp32(const e2m1_t* a, const e2m1_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp32a_e2m1_fp16(const e2m1_t* a, const e2m1_t* b, size_t len, fp16_t c);
```

#### MXFP4 — DP64A with UE8M0 block scaling

**Operation**

$$D = \sum_{g} \left( s^A_g \cdot s^B_g \cdot \sum_{i \in \text{block}_g} A_i \times B_i \right) + C$$

Each 64-element vector is divided into **2 blocks of 32 elements**. Each block carries one UE8M0 scale byte for A and one for B.

**Scale Format — UE8M0**

Each scale byte encodes a power-of-two multiplier: $2^{v - 127}$ for byte value $v$. Special cases: `0x00` → zero, `0xFF` → NaN (forces block result to NaN sentinel).

**Internal Representation**

Products and scale factors are accumulated using a **35-bit** mantissa intermediate (`Interm35`), wider than the unscaled path, to preserve precision across the scale multiplication.

**Supported Variants**

| Input type | Scale format | Block size | Accumulator | Rounding |
|------------|-------------|------------|-------------|----------|
| FP4 E2M1 | UE8M0 | 32 elements | FP32 | TC-Truncation |

**Public API**

```cpp
fp32_t blackwell_mxfp4_e2m1_e8_fp32(const e2m1_t* a, const e2m1_t* b,
                                      const uint8_t* sa, const uint8_t* sb,
                                      size_t len, fp32_t c);
```

#### NVFP4 — DP64A with UE4M3 block scaling

**Operation**

Same structure as MXFP4, but each 64-element vector is divided into **4 blocks of 16 elements**, and scale factors use the UE4M3 format.

**Scale Format — UE4M3**

Each scale byte (7 bits used) encodes a floating-point multiplier with a 4-bit exponent and 3-bit mantissa (bias 10 for normal values, bias 9 for subnormal). Special cases: `0x00` → zero, `0x7F`/`0xFF` → NaN.

**Internal Representation**

Same 35-bit mantissa intermediate as MXFP4 (`Interm35`).

**Supported Variants**

| Input type | Scale format | Block size | Accumulator | Rounding |
|------------|-------------|------------|-------------|----------|
| FP4 E2M1 | UE4M3 | 16 elements | FP32 | TC-Truncation |

**Public API**

```cpp
fp32_t blackwell_nvfp4_e2m1_ue4m3_fp32(const e2m1_t* a, const e2m1_t* b,
                                         const uint8_t* sa, const uint8_t* sb,
                                         size_t len, fp32_t c);
```

---

### Custom — Fully Configurable DPnA

**Operation**

$$D = \sum_{i=0}^{n-1} A_i \times B_i + C$$

where $n$ is user-specified (any positive integer).

**Configuration Parameters**

| Parameter | Options | Description |
|-----------|---------|-------------|
| A/B precision | FP16, BF16, FP8 E5M2, FP8 E4M3, FP4 E2M1 | Input element format |
| C/D precision | FP32, FP16 | Accumulator format |
| DP width `n` | any ≥ 1 | Elements per dot-product group |
| Mantissa bits `w` | 1–52 | Internal significand width (w+1 bits total) |
| Rounding mode | RTZ (truncate), RNE | Final rounding to output format |
| Scaling | optional | Per-group scale factors for A and B |
| Scale group size `x` | any ≥ 1 | Elements sharing one scale factor |
| Scale format | UE8M0, UE4M3 | Scale factor encoding |

The discard threshold follows the same rule as the fixed architectures: a value right-shifted by $\geq w+1$ bits is discarded entirely.

**Scaling**

When scaling is enabled, each group of `x` consecutive elements shares a single scale factor:

- **UE8M0** — 8-bit unsigned integer exponent (used in MXFP microscaling); the effective multiplier is $2^{\text{byte}}$.
- **UE4M3** — 7-bit unsigned float with 4-bit exponent and 3-bit mantissa (used in NVFP); the effective multiplier is the decoded float value.

Scale factors are applied to each element before the dot-product accumulation.

**Public API**

```cpp
uint32_t custom_dot_product(const uint32_t* a, const uint32_t* b,
                             const uint8_t* sa, const uint8_t* sb,
                             size_t len, uint32_t c,
                             const CustomConfig& cfg);
```

**`CustomConfig` struct** (defined in `include/custom_tensor.h`):

```cpp
struct CustomConfig {
    enum class ABPrec    : uint8_t { FP16, BF16, FP8_E5M2, FP8_E4M3, FP4_E2M1 };
    enum class CDPrec    : uint8_t { FP32, FP16 };
    enum class RoundMode : uint8_t { RNE, RTZ };
    enum class ScaleType : uint8_t { UE8M0, UE4M3 };

    ABPrec    ab_prec    = ABPrec::FP16;
    CDPrec    cd_prec    = CDPrec::FP32;
    int       dp_width   = 16;
    int       mant_width = 25;
    RoundMode round_mode = RoundMode::RTZ;
    bool      use_scale  = false;
    int       scale_group = 16;
    ScaleType scale_type = ScaleType::UE8M0;
    int       vec_len    = 0;
};
```

---

## Project Structure

```
tensor_sim/
├── CMakeLists.txt
├── main.cpp                       # Interactive CLI entry point (tensor_sim)
├── include/
│   ├── fp_utils.h                 # Bit-manipulation utilities for FP16 / BF16 / FP8 / FP4 / FP32
│   ├── volta_tensor.h             # Volta DP4A declarations
│   ├── ampere_tensor.h            # Ampere DP8A declarations
│   ├── hopper_tensor.h            # Hopper DP16A / DP32A declarations
│   ├── blackwell_tensor.h         # Blackwell DP16A / DP32A declarations
│   └── custom_tensor.h            # Custom DPnA declarations and CustomConfig struct
├── src/
│   ├── volta_tensor.cpp           # Volta DP4A implementation
│   ├── ampere_tensor.cpp          # Ampere DP8A implementation
│   ├── hopper_tensor.cpp          # Hopper DP16A / DP32A implementation
│   ├── blackwell_tensor.cpp       # Blackwell DP16A / DP32A implementation
│   └── custom_tensor.cpp          # Custom DPnA implementation
├── tests/
│   ├── test_volta.cpp                  # Volta unit tests
│   ├── test_ampere.cpp                 # Ampere unit tests
│   ├── test_hopper.cpp                 # Hopper unit tests
│   ├── test_blackwell.cpp              # Blackwell unit tests (incl. DP32A E2M1, MXFP4, NVFP4)
│   ├── test_fp32.fp16.txt              # Test vectors: FP16 inputs / FP32 accumulator
│   ├── test_fp16.fp16.txt              # Test vectors: FP16 inputs / FP16 accumulator
│   ├── test_fp32.bf16.txt              # Test vectors: BF16 inputs / FP32 accumulator
│   ├── test_fp32.fp8_e5m2.txt          # Test vectors: FP8 E5M2 inputs / FP32 accumulator
│   ├── test_fp16.fp8_e5m2.txt          # Test vectors: FP8 E5M2 inputs / FP16 accumulator
│   ├── test_fp32.fp8_e4m3.txt          # Test vectors: FP8 E4M3 inputs / FP32 accumulator
│   ├── test_fp16.fp8_e4m3.txt          # Test vectors: FP8 E4M3 inputs / FP16 accumulator
│   ├── test_fp32.fp4_e2m1.txt          # Test vectors: FP4 E2M1 inputs / FP32 accumulator (DP32A)
│   ├── test_fp16.fp4_e2m1.txt          # Test vectors: FP4 E2M1 inputs / FP16 accumulator (DP32A)
│   ├── test_fp32.fp4_e2m1_e8.txt       # Test vectors: MXFP4 (FP4 E2M1 + UE8M0 scales) / FP32
│   ├── test_fp32.fp4_e2m1_ue4m3.txt    # Test vectors: NVFP4 (FP4 E2M1 + UE4M3 scales) / FP32
│   └── test_custom.txt                 # Test vectors for the Custom DPnA mode
└── results/                            # Simulation output (generated at runtime)
```

---

## Building

**Requirements:** CMake ≥ 3.14, GCC/MinGW with C++17 support

```bash
cd tensor_sim
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Build artifacts:

| Target | Description |
|--------|-------------|
| `libvolta_sim.a`    | Volta static library |
| `libampere_sim.a`   | Ampere static library |
| `libhopper_sim.a`   | Hopper static library |
| `libblackwell_sim.a`| Blackwell static library |
| `libcustom_sim.a`   | Custom DPnA static library |
| `test_volta.exe`    | Volta unit test executable |
| `test_ampere.exe`   | Ampere unit test executable |
| `test_hopper.exe`   | Hopper unit test executable |
| `test_blackwell.exe`| Blackwell unit test executable |
| `tensor_sim.exe`    | Interactive simulator |

---

## Running Tests

```bash
cd build
./test_volta.exe
./test_ampere.exe
./test_hopper.exe
./test_blackwell.exe
```

Test coverage (all architectures):

- Basic integer dot-products and accumulations
- Special-value outputs: NaN, ±Inf, 0×∞, +∞+(−∞)
- Full subnormal input computation paths
- Negative zero / positive zero equivalence
- Multi-group chaining (vector length exceeds one group)
- RNE rounding (FP16 accumulator): tie-to-even cases
- FP8 E5M2 / E4M3 special values (E4M3 has no Inf encoding)
- FP4 E2M1 DP32A: basic products, subnormals, chaining
- MXFP4 / NVFP4: UE8M0 and UE4M3 scale factor handling, NaN scale propagation

---

## Interactive CLI Tool

Run `tensor_sim.exe` and follow the prompts. For architectures 1–4 (Volta / Ampere / Hopper / Blackwell) the program reads the corresponding test vector file and writes results to the `results/` directory. For architecture 5 (Custom) the prompts collect all configuration parameters and file paths interactively.

**Fixed-architecture session example (Blackwell, FP32 accumulator, MXFP4 inputs):**

```
=== Tensor Core Simulator ===

Select architecture:
  1. Volta
  2. Ampere
  3. Hopper
  4. Blackwell
  5. Custom (fully configurable)
Enter choice (1-5): 4

Select accumulator (C) precision:
  1. FP32
  2. FP16
Enter choice (1-2): 1

Select A/B vector precision:
  1. FP16
  2. BF16
  3. FP8_E5M2
  4. FP8_E4M3
  5. FP4_E2M1 (no scale)
  6. FP4_E2M1_E8 (MXFP4, E8 scale)
  7. FP4_E2M1_UE4M3 (NVFP4, UE4M3 scale)
Enter choice (1-7): 6
```

**Custom-architecture session example:**

```
=== Tensor Core Simulator ===

Select architecture:
  1. Volta
  2. Ampere
  3. Hopper
  4. Blackwell
  5. Custom (fully configurable)
Enter choice (1-5): 5

Select A/B input precision:
  1. FP16
  2. BF16
  3. FP8 E5M2
  4. FP8 E4M3
  5. FP4 E2M1
Enter choice (1-5): 3

Select C/D accumulator precision:
  1. FP32
  2. FP16
Enter choice (1-2): 1

Dot-product unit width n (e.g. 4, 8, 16, 32, 64): 32
Intermediate mantissa bits w (1-52): 20

Rounding mode:
  1. RTZ (round toward zero / truncate)
  2. RNE (round to nearest even)
Enter choice (1-2): 2

Use scaling factors for A/B? (0=no, 1=yes): 1
Elements per scale group x: 16
Scale factor format:
  1. UE8M0 (8-bit unsigned exponent, used in MXFP)
  2. UE4M3 (7-bit unsigned float, used in NVFP)
Enter choice (1-2): 1
Vector length per test line (A/B elements per line): 32

Test input file path: tests/my_custom.txt
Result output file path: results/my_custom_out.txt
```

**Fixed-architecture combinations (choices 1–4):**

| Architecture | A/B Precision | Accumulator | Input file | Output file |
|--------------|--------------|-------------|------------|-------------|
| Volta     | FP16     | FP32 | `tests/test_fp32.fp16.txt`     | `results/test_fp32.fp16_Volta.txt`          |
| Volta     | FP16     | FP16 | `tests/test_fp16.fp16.txt`     | `results/test_fp16.fp16_Volta.txt`          |
| Ampere    | FP16     | FP32 | `tests/test_fp32.fp16.txt`     | `results/test_fp32.fp16_Ampere.txt`         |
| Ampere    | FP16     | FP16 | `tests/test_fp16.fp16.txt`     | `results/test_fp16.fp16_Ampere.txt`         |
| Ampere    | BF16     | FP32 | `tests/test_fp32.bf16.txt`     | `results/test_fp32.bf16_Ampere.txt`         |
| Hopper    | FP16     | FP32 | `tests/test_fp32.fp16.txt`     | `results/test_fp32.fp16_Hopper.txt`         |
| Hopper    | FP16     | FP16 | `tests/test_fp16.fp16.txt`     | `results/test_fp16.fp16_Hopper.txt`         |
| Hopper    | BF16     | FP32 | `tests/test_fp32.bf16.txt`     | `results/test_fp32.bf16_Hopper.txt`         |
| Hopper    | FP8 E5M2 | FP32 | `tests/test_fp32.fp8_e5m2.txt` | `results/test_fp32.fp8_e5m2_Hopper.txt`    |
| Hopper    | FP8 E5M2 | FP16 | `tests/test_fp16.fp8_e5m2.txt` | `results/test_fp16.fp8_e5m2_Hopper.txt`    |
| Hopper    | FP8 E4M3 | FP32 | `tests/test_fp32.fp8_e4m3.txt` | `results/test_fp32.fp8_e4m3_Hopper.txt`    |
| Hopper    | FP8 E4M3 | FP16 | `tests/test_fp16.fp8_e4m3.txt` | `results/test_fp16.fp8_e4m3_Hopper.txt`    |
| Blackwell | FP16          | FP32 | `tests/test_fp32.fp16.txt`              | `results/test_fp32.fp16_Blackwell.txt`             |
| Blackwell | FP16          | FP16 | `tests/test_fp16.fp16.txt`              | `results/test_fp16.fp16_Blackwell.txt`             |
| Blackwell | BF16          | FP32 | `tests/test_fp32.bf16.txt`              | `results/test_fp32.bf16_Blackwell.txt`             |
| Blackwell | FP8 E5M2      | FP32 | `tests/test_fp32.fp8_e5m2.txt`          | `results/test_fp32.fp8_e5m2_Blackwell.txt`         |
| Blackwell | FP8 E5M2      | FP16 | `tests/test_fp16.fp8_e5m2.txt`          | `results/test_fp16.fp8_e5m2_Blackwell.txt`         |
| Blackwell | FP8 E4M3      | FP32 | `tests/test_fp32.fp8_e4m3.txt`          | `results/test_fp32.fp8_e4m3_Blackwell.txt`         |
| Blackwell | FP8 E4M3      | FP16 | `tests/test_fp16.fp8_e4m3.txt`          | `results/test_fp16.fp8_e4m3_Blackwell.txt`         |
| Blackwell | FP4 E2M1      | FP32 | `tests/test_fp32.fp4_e2m1.txt`          | `results/test_fp32.fp4_e2m1_Blackwell.txt`         |
| Blackwell | FP4 E2M1      | FP16 | `tests/test_fp16.fp4_e2m1.txt`          | `results/test_fp16.fp4_e2m1_Blackwell.txt`         |
| Blackwell | MXFP4 (E2M1+UE8M0) | FP32 | `tests/test_fp32.fp4_e2m1_e8.txt`   | `results/test_fp32.fp4_e2m1_e8_Blackwell.txt`      |
| Blackwell | NVFP4 (E2M1+UE4M3) | FP32 | `tests/test_fp32.fp4_e2m1_ue4m3.txt`| `results/test_fp32.fp4_e2m1_ue4m3_Blackwell.txt`   |

**Custom architecture (choice 5):** input file path and output file path are specified interactively. Any combination of supported precisions, DP width, mantissa bits, rounding mode, and optional scaling is accepted.

---

## Test Vector File Format

Each line is one test case. Fields are comma-separated hex values. Blank lines are ignored. The simulator appends the computed result `D` as a trailing hex field, e.g. `, 0x3F800000` (FP32) or `, 0x3C00` (FP16).

**Without scaling** (Volta / Ampere / Hopper / Blackwell DP16A–DP32A / Custom without scaling):

```
A[0], A[1], ..., A[L-1], B[0], B[1], ..., B[L-1], C
```

- `A` and `B`: element bit-patterns. Width per element: 16-bit for FP16/BF16, 8-bit for FP8, 4-bit for FP4 stored as 8-bit values.
- `C`: accumulator bit-pattern (32-bit for FP32, 16-bit for FP16).
- `L`: total vector length; groups of `n` are formed automatically with zero-padding for incomplete final groups.

**With scaling — MXFP4** (`fp4_e2m1_e8`, Blackwell only):

```
A[0], ..., A[L-1], B[0], ..., B[L-1], SA[0], SA[1], SB[0], SB[1], C
```

- `L` = 64 (one 64-element group); the group is split into two 32-element blocks.
- `SA[0]`, `SA[1]`: UE8M0 scale bytes for block 0 and block 1 of A.
- `SB[0]`, `SB[1]`: UE8M0 scale bytes for block 0 and block 1 of B.
- Always FP32 accumulator (32-bit C).

**With scaling — NVFP4** (`fp4_e2m1_ue4m3`, Blackwell only):

```
A[0], ..., A[L-1], B[0], ..., B[L-1], SA[0], SA[1], SA[2], SA[3], SB[0], SB[1], SB[2], SB[3], C
```

- `L` = 64 (one 64-element group); the group is split into four 16-element blocks.
- `SA[0..3]`: UE4M3 scale bytes for blocks 0–3 of A.
- `SB[0..3]`: UE4M3 scale bytes for blocks 0–3 of B.
- Always FP32 accumulator (32-bit C).

**With scaling — Custom** (`use_scale = true`):

```
A[0], ..., A[L-1], B[0], ..., B[L-1], SA[0], ..., SA[K-1], SB[0], ..., SB[K-1], C
```

- `K = ceil(L / x)` where `x` is the configured scale group size.
- Scale bytes are UE8M0 or UE4M3 depending on the configured scale type.

---
