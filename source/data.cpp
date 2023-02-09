#include "data.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>

Data::Data() {
    memset(in, 0, sizeof(in));
    out = -1;
}
void Data::normalize() {
    int min_i = 32, min_j = 32, max_i = -1, max_j = -1;

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            if (in[i][j] > 0) {
                min_i = std::min(min_i, i);
                max_i = std::max(max_i, i);
                min_j = std::min(min_j, j);
                max_j = std::max(max_j, j);
            }
    if (max_i == -1)
        return;
    cv::Mat mat(max_i - min_i + 1, max_j - min_j + 1, CV_8UC1);
    for (int i = min_i; i <= max_i; i++)
        for (int j = min_j; j <= max_j; j++)
            mat.at<uchar>(i - min_i, j - min_j) = in[i][j] * 255 + 0.5;
    cv::resize(mat, mat, cv::Size(32, 32));
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            in[i][j] = mat.at<uchar>(i, j) / 255.f;
}
void Data::add_noise(float m) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++) {
            in[i][j] += randf(m);
            in[i][j] = std::max(in[i][j], 0.f);
            in[i][j] = std::min(in[i][j], 1.f);
        }
}

int Dataset::read_int(FILE *f) {
    int x = 0;
    x = x << 8 | fgetc(f);
    x = x << 8 | fgetc(f);
    x = x << 8 | fgetc(f);
    x = x << 8 | fgetc(f);
    return x;
}
Dataset::Dataset(const char *img_path, const char *lab_path) {
    FILE *img_f = fopen(img_path, "rb"), *lab_f = fopen(lab_path, "rb");

    if (img_f == NULL) {
        fprintf(stderr, "Error: opening image file %s failed\n", img_path);
        exit(-3);
    }
    if (lab_f == NULL) {
        fprintf(stderr, "Error: opening label file %s failed\n", lab_path);
        exit(-4);
    }

    read_int(img_f);
    read_int(lab_f);
    read_int(img_f);
    n = read_int(lab_f);

    r = read_int(img_f);
    c = read_int(img_f);

    datas = new Data[n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < r; j++)
            for (int k = 0; k < c; k++)
                datas[i].in[j + (32 - r) / 2][k + (32 - c) / 2] =
                    fgetc(img_f) / 255.0;
        datas[i].normalize();
        datas[i].out = fgetc(lab_f);
    }
    fclose(img_f);
    fclose(lab_f);
}
Dataset::Dataset(Dataset &b, int l, int _n) {
    n = _n;
    datas = new Data[n];
    r = b.r;
    c = b.c;
    for (int i = 0; i < n; i++)
        datas[i] = b.datas[l + i];
}
Dataset::~Dataset() { delete[] datas; }

Data &Dataset::operator[](int i) { return datas[i]; }

void Dataset::add_noise(float m) {
    for (int i = 0; i < n; i++)
        datas[i].add_noise(m);
}