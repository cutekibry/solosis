#ifndef DATA_H
#define DATA_H

#include "maths.h"
#include <vector>
#include <cstdio>

struct Data
{
    std::vector<unsigned char> input;
    char output;
};
typedef std::vector<Data> Dataset;

Dataset read_dataset(const char *label_file, const char *image_file, int &r, int &c);

#endif