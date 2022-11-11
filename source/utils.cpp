#include "utils.h"
#include <string>
#include <filesystem>

void create_parents_for_file(const char *file_path) {
    std::string s = file_path;
    s = s.substr(0, s.rfind('/'));
    std::filesystem::create_directories(std::filesystem::path(s));
}