#include "data.h"
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>

Data::Data() {}
Data::Data(const char *img_path) {
    cv::Mat img = cv::imread(img_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::resize(img, img, cv::Size(28, 28));
    in = new float[28 * 28];
    for(int i=0; i<28; i++) for(int j=0; j<28; j++) in[i * 28 + j] = img.at<uchar>(i, j) / 255.0;
    out = -1;
}
void Data::save_img(const char *save_path) {
    cv::Mat img(28, 28, CV_8UC1);
    for (int i = 0; i < 28; i++)
        for (int j = 0; j < 28; j++)
            img.at<uchar>(i, j) = in[i * 28 + j] * 255 + 0.5;
    cv::imwrite(save_path, img);
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
    if(lab_f == NULL) {
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
        datas[i].in = new float[r * c];
        for (int j = 0; j < r * c; j++)
            datas[i].in[j] = fgetc(img_f) / 255.0;
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
    for(int i=0; i<n; i++)
        datas[i] = b.datas[l + i];
}
Dataset::~Dataset() {
    delete[] datas;
}

Data &Dataset::operator[](int i) { return datas[i]; }