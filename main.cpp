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
#include "custom_tensor.h"

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

    std::cout << "=== Tensor Core Simulator ===\n\n";
    std::cout << "Select architecture:\n"
              << "  1. Volta\n"
              << "  2. Ampere\n"
              << "  3. Hopper\n"
              << "  4. Blackwell\n"
              << "  5. Custom (fully configurable)\n"
              << "Enter choice (1-5): ";
    int arch_choice = 0;
    std::cin >> arch_choice;

    // -----------------------------------------------------------------------
    // Custom architecture path: collect config and run directly
    // -----------------------------------------------------------------------
    if (arch_choice == 5) {
        CustomConfig cfg;

        std::cout << "\nSelect A/B input precision:\n"
                  << "  1. FP16\n"
                  << "  2. BF16\n"
                  << "  3. FP8 E5M2\n"
                  << "  4. FP8 E4M3\n"
                  << "  5. FP4 E2M1\n"
                  << "Enter choice (1-5): ";
        int ab_ch = 0; std::cin >> ab_ch;
        switch (ab_ch) {
            case 1: cfg.ab_prec = CustomConfig::ABPrec::FP16;     break;
            case 2: cfg.ab_prec = CustomConfig::ABPrec::BF16;     break;
            case 3: cfg.ab_prec = CustomConfig::ABPrec::FP8_E5M2; break;
            case 4: cfg.ab_prec = CustomConfig::ABPrec::FP8_E4M3; break;
            case 5: cfg.ab_prec = CustomConfig::ABPrec::FP4_E2M1; break;
            default: std::cerr << "Invalid.\n"; return 1;
        }

        std::cout << "\nSelect C/D accumulator precision:\n"
                  << "  1. FP32\n"
                  << "  2. FP16\n"
                  << "Enter choice (1-2): ";
        int cd_ch = 0; std::cin >> cd_ch;
        switch (cd_ch) {
            case 1: cfg.cd_prec = CustomConfig::CDPrec::FP32; break;
            case 2: cfg.cd_prec = CustomConfig::CDPrec::FP16; break;
            default: std::cerr << "Invalid.\n"; return 1;
        }

        std::cout << "\nDot-product unit width n (e.g. 4, 8, 16, 32, 64): ";
        std::cin >> cfg.dp_width;
        if (cfg.dp_width < 1) { std::cerr << "n must be >= 1.\n"; return 1; }

        std::cout << "Intermediate mantissa bits w (1-52): ";
        std::cin >> cfg.mant_width;
        if (cfg.mant_width < 1 || cfg.mant_width > 52) {
            std::cerr << "w must be 1-52.\n"; return 1;
        }

        std::cout << "\nRounding mode:\n"
                  << "  1. RTZ (round toward zero / truncate)\n"
                  << "  2. RNE (round to nearest even)\n"
                  << "Enter choice (1-2): ";
        int rm_ch = 0; std::cin >> rm_ch;
        switch (rm_ch) {
            case 1: cfg.round_mode = CustomConfig::RoundMode::RTZ; break;
            case 2: cfg.round_mode = CustomConfig::RoundMode::RNE; break;
            default: std::cerr << "Invalid.\n"; return 1;
        }

        std::cout << "\nUse scaling factors for A/B? (0=no, 1=yes): ";
        int use_sc = 0; std::cin >> use_sc;
        cfg.use_scale = (use_sc != 0);

        if (cfg.use_scale) {
            std::cout << "Elements per scale group x: ";
            std::cin >> cfg.scale_group;
            if (cfg.scale_group < 1) { std::cerr << "x must be >= 1.\n"; return 1; }

            std::cout << "Scale factor format:\n"
                      << "  1. UE8M0 (8-bit unsigned exponent, used in MXFP)\n"
                      << "  2. UE4M3 (7-bit unsigned float, used in NVFP)\n"
                      << "Enter choice (1-2): ";
            int sf_ch = 0; std::cin >> sf_ch;
            switch (sf_ch) {
                case 1: cfg.scale_type = CustomConfig::ScaleType::UE8M0; break;
                case 2: cfg.scale_type = CustomConfig::ScaleType::UE4M3; break;
                default: std::cerr << "Invalid.\n"; return 1;
            }

            std::cout << "Vector length per test line (A/B elements per line): ";
            std::cin >> cfg.vec_len;
            if (cfg.vec_len < 1) { std::cerr << "vec_len must be >= 1.\n"; return 1; }
        }

        std::cout << "\nTest input file path: ";
        std::string input_path;
        std::cin >> input_path;

        std::cout << "Result output file path: ";
        std::string output_path;
        std::cin >> output_path;

        std::cout << "\n=== Custom Configuration ===\n"
                  << "  A/B precision : "
                  << (ab_ch==1?"FP16":ab_ch==2?"BF16":ab_ch==3?"FP8_E5M2":ab_ch==4?"FP8_E4M3":"FP4_E2M1") << "\n"
                  << "  C/D precision : " << (cd_ch==1?"FP32":"FP16") << "\n"
                  << "  DP width n    : " << cfg.dp_width << "\n"
                  << "  Mant bits w   : " << cfg.mant_width << "\n"
                  << "  Rounding      : " << (rm_ch==1?"RTZ":"RNE") << "\n"
                  << "  Scale         : " << (cfg.use_scale?"yes":"no");
        if (cfg.use_scale)
            std::cout << "  (group=" << cfg.scale_group
                      << ", type=" << (cfg.scale_type==CustomConfig::ScaleType::UE8M0?"UE8M0":"UE4M3") << ")";
        std::cout << "\n  Input         : " << input_path
                  << "\n  Output        : " << output_path << "\n\n";

        std::ifstream fin(input_path);
        if (!fin.is_open()) { std::cerr << "Cannot open: " << input_path << "\n"; return 1; }
        fs::create_directories(fs::path(output_path).parent_path());
        std::ofstream fout(output_path);
        if (!fout.is_open()) { std::cerr << "Cannot open: " << output_path << "\n"; return 1; }

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
                if (!t.empty()) vals.push_back(parse_hex(t));
            }
            if (vals.size() < 3) {
                std::cerr << "Warning: line " << line_num << " has fewer than 3 values, skipping.\n";
                continue;
            }

            uint32_t c_raw = vals.back();
            size_t n_ab    = vals.size() - 1;   // everything except C

            size_t L, K = 0;
            if (!cfg.use_scale) {
                L = n_ab / 2;
            } else {
                L = (size_t)cfg.vec_len;
                K = ((size_t)cfg.vec_len + cfg.scale_group - 1) / cfg.scale_group;
                if (n_ab < 2 * L + 2 * K) {
                    std::cerr << "Warning: line " << line_num << " too short, skipping.\n";
                    continue;
                }
            }

            std::vector<uint32_t> A(L), B(L);
            for (size_t k = 0; k < L; ++k) A[k] = vals[k];
            for (size_t k = 0; k < L; ++k) B[k] = vals[L + k];

            std::vector<uint8_t> sa(K), sb(K);
            if (cfg.use_scale) {
                size_t base = 2 * L;
                for (size_t k = 0; k < K; ++k) sa[k] = (uint8_t)vals[base + k];
                for (size_t k = 0; k < K; ++k) sb[k] = (uint8_t)vals[base + K + k];
            }

            uint32_t result = custom_dot_product(
                A.data(), B.data(),
                cfg.use_scale ? sa.data() : nullptr,
                cfg.use_scale ? sb.data() : nullptr,
                L, c_raw, cfg);

            fout << stripped << ", 0x";
            if (cfg.cd_prec == CustomConfig::CDPrec::FP32)
                fout << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << result << "\n";
            else
                fout << std::uppercase << std::hex << std::setw(4) << std::setfill('0') << (result & 0xFFFFu) << "\n";
        }

        std::cout << "Done. Processed " << std::dec << line_num << " test case(s).\n"
                  << "Results written to: " << output_path << "\n";
        return 0;
    }

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
              << "  5. FP4_E2M1 (no scale)\n"
              << "  6. FP4_E2M1_E8 (MXFP4, E8 scale)\n"
              << "  7. FP4_E2M1_UE4M3 (NVFP4, UE4M3 scale)\n"
              << "Enter choice (1-7): ";
    int ab_choice = 0;
    std::cin >> ab_choice;

    std::string ab_type;
    switch (ab_choice) {
        case 1: ab_type = "fp16";             break;
        case 2: ab_type = "bf16";             break;
        case 3: ab_type = "fp8_e5m2";         break;
        case 4: ab_type = "fp8_e4m3";         break;
        case 5: ab_type = "fp4_e2m1";         break;
        case 6: ab_type = "fp4_e2m1_e8";      break;
        case 7: ab_type = "fp4_e2m1_ue4m3";   break;
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
    if (arch_name == "Blackwell" && ab_type == "fp4_e2m1")                     implemented = true;
    if (arch_name == "Blackwell" && ab_type == "fp4_e2m1_e8"    && c_type == "fp32") implemented = true;
    if (arch_name == "Blackwell" && ab_type == "fp4_e2m1_ue4m3" && c_type == "fp32") implemented = true;

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
        const uint32_t c_raw = vals.back();

        size_t vec_len = 0;
        size_t n_sa = 0, n_sb = 0;
        if (ab_type == "fp4_e2m1_e8") {
            n_sa = 2; n_sb = 2;
            vec_len = (n_ab - n_sa - n_sb) / 2;
        } else if (ab_type == "fp4_e2m1_ue4m3") {
            n_sa = 4; n_sb = 4;
            vec_len = (n_ab - n_sa - n_sb) / 2;
        } else {
            vec_len = n_ab / 2;
        }

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

        } else if (arch_name == "Blackwell" && ab_type == "fp4_e2m1") {
            std::vector<e2m1_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<e2m1_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<e2m1_t>(vals[vec_len + i]);

            if (c_type == "fp32") {
                fp32_t result = blackwell_dp32a_e2m1_fp32(A.data(), B.data(), vec_len,
                                                           static_cast<fp32_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(8) << std::setfill('0') << result << "\n";
            } else {
                fp16_t result = blackwell_dp32a_e2m1_fp16(A.data(), B.data(), vec_len,
                                                           static_cast<fp16_t>(c_raw));
                fout << ", 0x" << std::uppercase << std::hex
                     << std::setw(4) << std::setfill('0') << result << "\n";
            }

        } else if (arch_name == "Blackwell" && ab_type == "fp4_e2m1_e8") {
            std::vector<e2m1_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<e2m1_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<e2m1_t>(vals[vec_len + i]);
            size_t scale_base = 2 * vec_len;
            std::vector<uint8_t> sa(n_sa), sb(n_sb);
            for (size_t i = 0; i < n_sa; ++i) sa[i] = static_cast<uint8_t>(vals[scale_base + i]);
            for (size_t i = 0; i < n_sb; ++i) sb[i] = static_cast<uint8_t>(vals[scale_base + n_sa + i]);

            fp32_t result = blackwell_mxfp4_e2m1_e8_fp32(A.data(), B.data(), sa.data(), sb.data(), vec_len,
                                                          static_cast<fp32_t>(c_raw));
            fout << ", 0x" << std::uppercase << std::hex
                 << std::setw(8) << std::setfill('0') << result << "\n";

        } else if (arch_name == "Blackwell" && ab_type == "fp4_e2m1_ue4m3") {
            std::vector<e2m1_t> A(vec_len), B(vec_len);
            for (size_t i = 0; i < vec_len; ++i) A[i] = static_cast<e2m1_t>(vals[i]);
            for (size_t i = 0; i < vec_len; ++i) B[i] = static_cast<e2m1_t>(vals[vec_len + i]);
            size_t scale_base = 2 * vec_len;
            std::vector<uint8_t> sa(n_sa), sb(n_sb);
            for (size_t i = 0; i < n_sa; ++i) sa[i] = static_cast<uint8_t>(vals[scale_base + i]);
            for (size_t i = 0; i < n_sb; ++i) sb[i] = static_cast<uint8_t>(vals[scale_base + n_sa + i]);

            fp32_t result = blackwell_nvfp4_e2m1_ue4m3_fp32(A.data(), B.data(), sa.data(), sb.data(), vec_len,
                                                             static_cast<fp32_t>(c_raw));
            fout << ", 0x" << std::uppercase << std::hex
                 << std::setw(8) << std::setfill('0') << result << "\n";
        }
    }

    std::cout << "Done. Processed " << std::dec << line_num << " test case(s).\n"
              << "Results written to: " << output_path << "\n";
    return 0;
}