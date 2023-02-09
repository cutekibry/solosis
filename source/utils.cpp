#include "utils.h"
#include <cmath>
#include <filesystem>
#include <string>

void create_parents_for_file(const char *file_path) {
    std::string s = file_path;
    s = s.substr(0, s.rfind('/'));
    std::filesystem::create_directories(std::filesystem::path(s));
}

float randf(float n) { return n - 2 * n * rand() / RAND_MAX; }

float sigma(float x) { return (x >= 0) ? x : (0.01 * x); }
float sigma_d(float x) { return (x >= 0) ? 1 : 0.01; }
float randarg(int in_size, int out_size) { return randf(sqrtf(6.0 / in_size)); }