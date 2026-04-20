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
  - [Blackwell — DP16A / DP32A](#blackwell--dp16a--dp32a)
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
| Blackwell | DP16A (16-point dot-product accumulate)| 16 | FP16     | FP32 | TC-Truncation |
| Blackwell | DP16A (16-point dot-product accumulate)| 16 | FP16     | FP16 | RNE |
| Blackwell | DP16A (16-point dot-product accumulate)| 16 | BF16     | FP32 | TC-Truncation |
| Blackwell | DP32A (32-point dot-product accumulate)| 32 | FP8 E5M2 | FP32 | TC-Truncation |
| Blackwell | DP32A (32-point dot-product accumulate)| 32 | FP8 E5M2 | FP16 | RNE |
| Blackwell | DP32A (32-point dot-product accumulate)| 32 | FP8 E4M3 | FP32 | TC-Truncation |
| Blackwell | DP32A (32-point dot-product accumulate)| 32 | FP8 E4M3 | FP16 | RNE |

---

## Floating-Point Format Reference

| Format     | Bits | Sign | Exponent (bias) | Mantissa | Max normal value  | NaN / Inf |
|------------|------|------|-----------------|----------|-------------------|-----------|
| FP32       | 32   | 1    | 8 (bias 127)    | 23       | ≈ 3.40 × 10³⁸    | Yes |
| FP16       | 16   | 1    | 5 (bias 15)     | 10       | 65504             | Yes |
| BF16       | 16   | 1    | 8 (bias 127)    |  7       | ≈ 3.39 × 10³⁸    | Yes |
| FP8 E5M2   |  8   | 1    | 5 (bias 15)     |  2       | 57344             | Yes |
| FP8 E4M3   |  8   | 1    | 4 (bias 7)      |  3       | 448               | NaN only (no Inf) |
| NVFP4 E2M1 |  4   | 1    | 2 (bias 1)      |  1       | 6.0               | None (planned) |

> **FP8 E4M3 special convention (NVIDIA)**: only `0x7F` (`+NaN`) and `0xFF` (`−NaN`) encode NaN; there is no Inf encoding; maximum normal value is 448.

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
// FP32 accumulator
fp32_t volta_dp4a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);

// FP16 accumulator
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
// FP16 inputs, FP32 accumulator
fp32_t ampere_dp8a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);

// FP16 inputs, FP16 accumulator
fp16_t ampere_dp8a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c);

// BF16 inputs, FP32 accumulator
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

### Blackwell — DP16A / DP32A

Blackwell's DP16A is identical to Hopper's (same group size, same 25-bit mantissa). The key upgrade is in DP32A: the internal mantissa precision is raised from Hopper's **13 bits** to **25 bits**, significantly improving FP8 dot-product accumulation accuracy.

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

---

## Project Structure

```
tensor_sim/
├── CMakeLists.txt
├── main.cpp                       # Interactive CLI entry point (tensor_sim)
├── include/
│   ├── fp_utils.h                 # Bit-manipulation utilities for FP16 / BF16 / FP8 / FP32
│   ├── volta_tensor.h             # Volta DP4A declarations
│   ├── ampere_tensor.h            # Ampere DP8A declarations
│   ├── hopper_tensor.h            # Hopper DP16A / DP32A declarations
│   └── blackwell_tensor.h         # Blackwell DP16A / DP32A declarations
├── src/
│   ├── volta_tensor.cpp           # Volta DP4A implementation
│   ├── ampere_tensor.cpp          # Ampere DP8A implementation
│   ├── hopper_tensor.cpp          # Hopper DP16A / DP32A implementation
│   └── blackwell_tensor.cpp       # Blackwell DP16A / DP32A implementation
├── tests/
│   ├── test_volta.cpp             # Volta unit tests
│   ├── test_ampere.cpp            # Ampere unit tests
│   ├── test_hopper.cpp            # Hopper unit tests
│   ├── test_blackwell.cpp         # Blackwell unit tests
│   ├── test_fp32.fp16.txt         # Test vectors: FP16 inputs / FP32 accumulator
│   ├── test_fp16.fp16.txt         # Test vectors: FP16 inputs / FP16 accumulator
│   ├── test_fp32.bf16.txt         # Test vectors: BF16 inputs / FP32 accumulator
│   ├── test_fp32.fp8_e5m2.txt     # Test vectors: FP8 E5M2 inputs / FP32 accumulator
│   ├── test_fp16.fp8_e5m2.txt     # Test vectors: FP8 E5M2 inputs / FP16 accumulator
│   ├── test_fp32.fp8_e4m3.txt     # Test vectors: FP8 E4M3 inputs / FP32 accumulator
│   └── test_fp16.fp8_e4m3.txt     # Test vectors: FP8 E4M3 inputs / FP16 accumulator
└── results/                       # Simulation output (generated at runtime)
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
| `test_volta.exe`    | Volta unit test executable |
| `test_ampere.exe`   | Ampere unit test executable |
| `test_hopper.exe`   | Hopper unit test executable |
| `test_blackwell.exe`| Blackwell unit test executable |
| `tensor_sim.exe`   | Interactive simulator |

---

## Running Tests

```bash
cd build
./test_volta.exe     # expected: 74 passed, 0 failed
./test_ampere.exe    # expected: 50 passed, 0 failed
./test_hopper.exe    # expected: 99 passed, 0 failed
./test_blackwell.exe # expected: 74 passed, 0 failed
```

Test coverage (all architectures):

- Basic integer dot-products and accumulations
- Special-value outputs: NaN, ±Inf, 0×∞, +∞+(−∞)
- Full subnormal input computation paths
- Negative zero / positive zero equivalence
- Multi-group chaining (vector length exceeds one group)
- RNE rounding (FP16 accumulator): tie-to-even cases
- FP8 E5M2 / E4M3 special values (E4M3 has no Inf encoding)

---

## Interactive CLI Tool

Run `tensor_sim.exe` and follow the prompts to select architecture, accumulator precision, and A/B vector precision. The program reads the corresponding test vector file, appends the simulation results, and writes output to the `results/` directory.

```
=== Tensor Core Simulator ===

Select architecture:
  1. Volta
  2. Ampere
  3. Hopper
  4. Blackwell
Enter choice (1-4): 3

Select accumulator (C) precision:
  1. FP32
  2. FP16
Enter choice (1-2): 1

Select A/B vector precision:
  1. FP16
  2. BF16
  3. FP8_E5M2
  4. FP8_E4M3
  5. NVFP4_E2M1
Enter choice (1-5): 3
```

**Currently implemented combinations:**

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
| Blackwell | FP16     | FP32 | `tests/test_fp32.fp16.txt`     | `results/test_fp32.fp16_Blackwell.txt`      |
| Blackwell | FP16     | FP16 | `tests/test_fp16.fp16.txt`     | `results/test_fp16.fp16_Blackwell.txt`      |
| Blackwell | BF16     | FP32 | `tests/test_fp32.bf16.txt`     | `results/test_fp32.bf16_Blackwell.txt`      |
| Blackwell | FP8 E5M2 | FP32 | `tests/test_fp32.fp8_e5m2.txt` | `results/test_fp32.fp8_e5m2_Blackwell.txt` |
| Blackwell | FP8 E5M2 | FP16 | `tests/test_fp16.fp8_e5m2.txt` | `results/test_fp16.fp8_e5m2_Blackwell.txt` |
| Blackwell | FP8 E4M3 | FP32 | `tests/test_fp32.fp8_e4m3.txt` | `results/test_fp32.fp8_e4m3_Blackwell.txt` |
| Blackwell | FP8 E4M3 | FP16 | `tests/test_fp16.fp8_e4m3.txt` | `results/test_fp16.fp8_e4m3_Blackwell.txt` |

---
