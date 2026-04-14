#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <algorithm>
#include <filesystem>

#include "volta_tensor.h"
#include "ampere_tensor.h"
#include "hopper_tensor.h"
#include "blackwell_tensor.h"

namespace fs = std::filesystem;

#ifndef TENSOR_SIM_DIR
#define TENSOR_SIM_DIR "."
#endif

static uint32_t parse_hex(const std::string& s) {
    return static_cast<uint32_t>(std::stoul(s, nullptr, 16));
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

int main() {

    std::cout << "=== TensorScope Tensor Core Simulator ===\n\n";
    std::cout << "Select architecture:\n"
              << "  1. Volta\n"
              << "  2. Ampere\n"
              << "  3. Hopper\n"
              << "  4. Blackwell\n"
              << "Enter choice (1-4): ";
    int arch_choice = 0;
    std::cin >> arch_choice;

    std::string arch_name;
    switch (arch_choice) {
        case 1: arch_name = "Volta";    break;
        case 2: arch_name = "Ampere";   break;
        case 3: arch_name = "Hopper";   break;
        case 4: arch_name = "Blackwell";break;
        default:
            std::cerr << "Invalid architecture choice.\n";
            return 1;
    }

    std::cout << "\nSelect accumulator (C) precision:\n"
              << "  1. FP32\n"
              << "  2. FP16\n"
              << "Enter choice (1-2): ";
    int c_choice = 0;
    std::cin >> c_choice;

    std::string c_type;
    switch (c_choice) {
        case 1: c_type = "fp32"; break;
        case 2: c_type = "fp16"; break;
        default:
            std::cerr << "Invalid accumulator precision choice.\n";
            return 1;
    }

    std::cout << "\nSelect A/B vector precision:\n"
              << "  1. FP16\n"
              << "  2. BF16\n"
              << "  3. FP8_E5M2\n"
              << "  4. FP8_E4M3\n"
              << "  5. NVFP4_E2M1\n"
              << "Enter choice (1-5): ";
    int ab_choice = 0;
    std::cin >> ab_choice;

    std::string ab_type;
    switch (ab_choice) {
        case 1: ab_type = "fp16";       break;
        case 2: ab_type = "bf16";       break;
        case 3: ab_type = "fp8_e5m2";   break;
        case 4: ab_type = "fp8_e4m3";   break;
        case 5: ab_type = "nvfp4_e2m1"; break;
        default:
            std::cerr << "Invalid A/B precision choice.\n";
            return 1;
    }

    const std::string base_dir    = TENSOR_SIM_DIR;
    const std::string stem        = "test_" + c_type + "." + ab_type;
    const std::string input_path  = base_dir + "/tests/"   + stem + ".txt";
    const std::string output_path = base_dir + "/results/" + stem + "_" + arch_name + ".txt";

    std::cout << "\nConfiguration summary:\n"
              << "  Architecture : " << arch_name  << "\n"
              << "  C precision  : " << c_type      << "\n"
              << "  A/B precision: " << ab_type     << "\n"
              << "  Input        : " << input_path  << "\n"
              << "  Output       : " << output_path << "\n\n";

    bool implemented = false;
    if (arch_name == "Volta"  && ab_type == "fp16")                      implemented = true;
    if (arch_name == "Ampere" && ab_type == "fp16")                      implemented = true;
    if (arch_name == "Ampere" && ab_type == "bf16" && c_type == "fp32")  implemented = true;
    if (arch_name == "Hopper" && ab_type == "fp16")                      implemented = true;
    if (arch_name == "Hopper" && ab_type == "bf16" && c_type == "fp32")  implemented = true;
    if (arch_name == "Hopper" && ab_type == "fp8_e5m2")                       implemented = true;
    if (arch_name == "Hopper" && ab_type == "fp8_e4m3")                       implemented = true;
    if (arch_name == "Blackwell" && ab_type == "fp16")                         implemented = true;
    if (arch_name == "Blackwell" && ab_type == "bf16" && c_type == "fp32")     implemented = true;
    if (arch_name == "Blackwell" && ab_type == "fp8_e5m2")                     implemented = true;
    if (arch_name == "Blackwell" && ab_type == "fp8_e4m3")                     implemented = true;

    if (!implemented) {
        std::cerr << "Error: " << arch_name << " + " << ab_type
                  << " is not yet implemented.\n";
        return 1;
    }

    std::ifstream fin(input_path);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open input file: " << input_path << "\n";
        return 1;
    }

    fs::create_directories(base_dir + "/results");
    std::ofstream fout(output_path);
    if (!fout.is_open()) {
        std::cerr << "Error: cannot open output file: " << output_path << "\n";
        return 1;
    }

    int line_num = 0;
    std::string line;
    while (std::getline(fin, line)) {
        std::string stripped = trim(line);
        if (stripped.empty()) continue;
        ++line_num;

        std::vector<uint32_t> vals;
        std::stringstream ss(stripped);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            std::string t = trim(tok);
            if (!t.empty())
                vals.push_back(parse_hex(t));
        }

        if (vals.size() < 3) {
            std::cerr << "Warning: line " << line_num
                      << " has fewer than 3 values, skipping.\n";
            continue;
        }

        const size_t n_ab    = vals.size() - 1;
        const size_t vec_len = n_ab / 2;
        const uint32_t c_raw = vals.back();

        fout << stripped;

        if (arch_name == "Volta") {
            std::vector<fp16_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<fp16_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<fp16_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = volta_dp4a_fp32(A.data(), B.data(), vec_len,
                                                static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = volta_dp4a_fp16(A.data(), B.data(), vec_len,
                                                static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }
        } else if (arch_name == "Ampere" && ab_type == "fp16") {
            std::vector<fp16_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<fp16_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<fp16_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = ampere_dp8a_fp32(A.data(), B.data(), vec_len,
                                                 static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = ampere_dp8a_fp16(A.data(), B.data(), vec_len,
                                                 static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }
        } else if (arch_name == "Ampere" && ab_type == "bf16") {
            std::vector<bf16_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<bf16_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<bf16_t>(vals[vec_len + i]);

            fp32_t result = ampere_dp8a_bf16(A.data(), B.data(), vec_len,
                                             static_cast<fp32_t>(c_raw));
            fout << ", 0x" << std::uppercase << std::hex
                 << std::setw(8) << std::setfill('0') << result << "\n";

        } else if (arch_name == "Hopper" && ab_type == "fp16") {
            std::vector<fp16_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<fp16_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<fp16_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = hopper_dp16a_fp32(A.data(), B.data(), vec_len,
                                                  static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = hopper_dp16a_fp16(A.data(), B.data(), vec_len,
                                                  static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }

        } else if (arch_name == "Hopper" && ab_type == "bf16") {
            std::vector<bf16_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<bf16_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<bf16_t>(vals[vec_len + i]);

            fp32_t result = hopper_dp16a_bf16(A.data(), B.data(), vec_len,
                                              static_cast<fp32_t>(c_raw));
            fout << ", 0x" << std::uppercase << std::hex
                 << std::setw(8) << std::setfill('0') << result << "\n";

        } else if (arch_name == "Hopper" && ab_type == "fp8_e5m2") {
            std::vector<e5m2_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<e5m2_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<e5m2_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = hopper_dp32a_e5m2_fp32(A.data(), B.data(), vec_len,
                                                        static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = hopper_dp32a_e5m2_fp16(A.data(), B.data(), vec_len,
                                                        static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }

        } else if (arch_name == "Hopper" && ab_type == "fp8_e4m3") {
            std::vector<e4m3_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<e4m3_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<e4m3_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = hopper_dp32a_e4m3_fp32(A.data(), B.data(), vec_len,
                                                        static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = hopper_dp32a_e4m3_fp16(A.data(), B.data(), vec_len,
                                                        static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }

        } else if (arch_name == "Blackwell" && ab_type == "fp16") {
            std::vector<fp16_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<fp16_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<fp16_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = blackwell_dp16a_fp32(A.data(), B.data(), vec_len,
                                                     static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = blackwell_dp16a_fp16(A.data(), B.data(), vec_len,
                                                     static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }

        } else if (arch_name == "Blackwell" && ab_type == "bf16") {
            std::vector<bf16_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<bf16_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<bf16_t>(vals[vec_len + i]);

            fp32_t result = blackwell_dp16a_bf16(A.data(), B.data(), vec_len,
                                                  static_cast<fp32_t>(c_raw));
            fout << ", 0x" << std::uppercase << std::hex
                 << std::setw(8) << std::setfill('0') << result << "\n";

        } else if (arch_name == "Blackwell" && ab_type == "fp8_e5m2") {
            std::vector<e5m2_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<e5m2_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<e5m2_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = blackwell_dp32a_e5m2_fp32(A.data(), B.data(), vec_len,
                                                           static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = blackwell_dp32a_e5m2_fp16(A.data(), B.data(), vec_len,
                                                           static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }

        } else if (arch_name == "Blackwell" && ab_type == "fp8_e4m3") {
            std::vector<e4m3_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<e4m3_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<e4m3_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = blackwell_dp32a_e4m3_fp32(A.data(), B.data(), vec_len,
                                                           static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = blackwell_dp32a_e4m3_fp16(A.data(), B.data(), vec_len,
                                                           static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }
        }
    }

    std::cout << "Done. Processed " << std::dec << line_num << " test case(s).\n"
              << "Results written to: " << output_path << "\n";
    return 0;
}