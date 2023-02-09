#include "predict.h"
#include "lenet.h"
#include "utils.h"
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace predict {
const int K = 10;
const int MAXDGT = 10000;
const char *MODEL_PATH = "model-data/standard.model";
const int DX = 6;
const int THRESHOLD = 130;

#define d_imshow(args...)                                                      \
    {                                                                          \
        if (DEBUG)                                                             \
            cv::imshow(args);                                                  \
    }

int h, w;
cv::Mat img;
cv::Mat gray_img;
cv::Mat vis_img;
Lenet *lenet;

bool DEBUG;

int max_i, min_i, max_j, min_j;

void dfs(int i, int j) {
    if (!vis_img.at<uchar>(i, j))
        return;
    max_i = std::max(max_i, i);
    min_i = std::min(min_i, i);
    max_j = std::max(max_j, j);
    min_j = std::min(min_j, j);
    vis_img.at<uchar>(i, j) = 0;
    for (int ii = std::max(i - DX, 0); ii <= std::min(i + DX, h - 1); ii++)
        for (int jj = std::max(j - DX, 0); jj <= std::min(j + DX, w - 1); jj++)
            dfs(ii, jj);
}
void cover() {
    for (int i = min_i; i <= max_i; i++)
        for (int j = min_j; j <= max_j; j++)
            vis_img.at<uchar>(i, j) = 0;
}
void normalize(cv::Mat &mat, int w, int h) {
    uchar maxv = 0, minv = 255;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            minv = std::min(minv, mat.at<uchar>(i, j));
            maxv = std::max(maxv, mat.at<uchar>(i, j));
        }
    if (minv != maxv)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                mat.at<uchar>(i, j) =
                    (mat.at<uchar>(i, j) - minv) * 255 / (maxv - minv) + 0.5f;
}
void add() {
    static float in[32][32];
    static cv::Mat dgt;
    static char tmp_c_str[2];

    cv::Rect rect =
        cv::Rect(min_j, min_i, max_j - min_j + 1, max_i - min_i + 1);

    dgt = gray_img(rect);
    d_imshow("Digit", dgt);

    cv::resize(dgt, dgt, cv::Size(32, 32));
    d_imshow("Resized digit", dgt);

    normalize(dgt, 32, 32);
    d_imshow("Final digit", dgt);

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            in[i][j] = dgt.at<uchar>(i, j) / 255.f;

    int val = lenet->predict(in);
    if (DEBUG)
        printf("# add (%d, %d) - (%d, %d): %d\n", min_i, min_j, max_i, max_j,
               val);

    tmp_c_str[0] = '0' + val;

    cv::rectangle(img, rect, cv::Scalar(0, 0, 255));
    cv::putText(img, cv::String(tmp_c_str), cv::Point(min_j, min_i),
                cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 255));
}

int main(int argc, char *argv[], void (*usage)(const char *prog)) {
    if (argc < 3) {
        usage(argv[0]);
        return -1;
    }

    if (argc >= 4 and strcmp(argv[3], "--debug") == 0)
        DEBUG = true;

    lenet = new Lenet(MODEL_PATH);
    lenet->toggle_test_mode(true);

    img = cv::imread(argv[2]);
    // cv::imshow("Original image", img);

    w = img.size().width;
    h = img.size().height;

    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    cv::bitwise_not(gray_img, gray_img);
    d_imshow("Gray-scaled inverted image", gray_img);

    normalize(gray_img, w, h);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            if (gray_img.at<uchar>(i, j) < THRESHOLD)
                gray_img.at<uchar>(i, j) = 0;
    d_imshow("Normalized image", gray_img);

    vis_img = gray_img.clone();

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (vis_img.at<uchar>(i, j)) {
                max_i = min_i = i;
                max_j = min_j = j;
                dfs(i, j);
                cover();
                add();
                if (DEBUG)
                    cv::waitKey(0);
            }
        }
    }

    cv::imshow("Result", img);
    cv::waitKey(0);

    return 0;
}
}; // namespace predict