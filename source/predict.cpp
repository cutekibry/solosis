#include "predict.h"
#include "netset.h"
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace predict {
const int N = 28;
const int K = 10;
const int MAXDGT = 100000;
const int THRESHOLD = 100;
const int DX = 6;

const int MAX_N0 = 20;
const int MIN_N0 = 10;

const int NETWORK_CNT = 1;
const char *NETWORK_PATH[NETWORK_CNT] = {
    "model-data/standard/256-128-64-32-10"};

#define d_imshow(args...)                                                         \
    {                                                                          \
        if (DEBUG)                                                             \
            cv::imshow(args);                                                   \
    }

int h, w;
cv::Mat img;
cv::Mat gray_img, canny;
Netset netset(NETWORK_CNT, K);

bool DEBUG;

struct Digit {
    int xl, xr, y;
    int val;
    Digit() {}
    Digit(int _xl, int _xr, int _y, int _val)
        : xl(_xl), xr(_xr), y(_y), val(_val) {}
};

Digit res[MAXDGT];
bool has_left[MAXDGT];
int right[MAXDGT];
int res_n;

int max_i, min_i, max_j, min_j;

bool on_right(Digit a, Digit b) {
    if (a.xr <= b.xr)
        return b.xl <= a.xr and a.y <= b.y;
    else
        return a.xl <= b.xr and a.y <= b.y;
}

void dfs(int i, int j) {
    if (!canny.at<uchar>(i, j))
        return;
    max_i = std::max(max_i, i);
    min_i = std::min(min_i, i);
    max_j = std::max(max_j, j);
    min_j = std::min(min_j, j);
    canny.at<uchar>(i, j) = 0;
    for (int ii = std::max(i - DX, 0); ii <= std::min(i + DX, h - 1); ii++)
        for (int jj = std::max(j - DX, 0); jj <= std::min(j + DX, w - 1); jj++)
            dfs(ii, jj);
}
void cover() {
    for (int i = min_i; i <= max_i; i++)
        for (int j = min_j; j <= max_j; j++)
            canny.at<uchar>(i, j) = 0;
}
void add() {
    static float in[N * N];
    static cv::Mat dgt, resize_dgt, filled_dgt;

    int dw = max_j - min_j + 1;
    int dh = max_i - min_i + 1;

    dgt = gray_img(cv::Rect(min_j, min_i, dw, dh));
    d_imshow("Digit", dgt);

    float p =
        std::max((float)MAX_N0 / std::max(dw, dh),
                 (float)std::min(std::min(MIN_N0, dw), dh) / std::min(dw, dh));
    dw = std::min(int(p * dw + 0.5), MAX_N0);
    dh = std::min(int(p * dh + 0.5), MAX_N0);
    cv::resize(dgt, resize_dgt, cv::Size(dw, dh), cv::INTER_CUBIC);
    d_imshow("Resized digit", resize_dgt);

    filled_dgt = cv::Mat::zeros(N, N, CV_8UC1);
    for (int i = 0; i < dh; i++)
        for (int j = 0; j < dw; j++) {
            uchar x = resize_dgt.at<uchar>(i, j);
            if (x >= THRESHOLD)
                filled_dgt.at<uchar>(i + (N - dh) / 2, j + (N - dw) / 2) = x;
        }
    // cv::imshow("Filled_Digit", filled_dgt);
    uchar maxv = -1;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            maxv = std::max(maxv, filled_dgt.at<uchar>(i, j));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            filled_dgt.at<uchar>(i, j) =
                (filled_dgt.at<uchar>(i, j) * 255.0 / maxv + 0.5);
    // filled_dgt.convertTo(filled_dgt, -1, 1.2, 0);
    d_imshow("Final digit", filled_dgt);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            in[i * N + j] = (float)filled_dgt.at<uchar>(i, j) / 255;

    int val = netset.predict(in);
    if (DEBUG)
        printf("# add (%d, %d) - (%d, %d): %d\n", min_i, min_j, max_i, max_j,
               val);
    has_left[res_n] = false;
    right[res_n] = -1;
    res[res_n] = Digit(min_i, max_i, min_j, val);
    res_n++;
}
int main(int argc, char *argv[], void (*usage)(const char *prog)) {
    if (argc < 3) {
        usage(argv[0]);
        return -1;
    }

    if (argc >= 4 and strcmp(argv[3], "--debug") == 0)
        DEBUG = true;

    for (int i = 0; i < NETWORK_CNT; i++)
        if (!netset.add_network(NETWORK_PATH[i])) {
            fprintf(stderr, "loading %s error\n", NETWORK_PATH[i]);
            return -2;
        }

    img = cv::imread(argv[2]);
    // cv::imshow("Original image", img);

    w = img.size().width;
    h = img.size().height;

    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    // cv::imshow("Gray-scaled image", gray_img);

    cv::bitwise_not(gray_img, gray_img);
    d_imshow("Gray-scaled inverted image", gray_img);

    // gray_img.convertTo(gray_img, -1, 1, 0);
    // d_imshow("Gray-scaled converted image", gray_img);

    cv::Canny(gray_img, canny, 10, 150);

    d_imshow("Canny-filtered image", canny);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (canny.at<uchar>(i, j)) {
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

    for (int i = 0; i < res_n; i++) {
        for (int j = 0; j < res_n; j++)
            if (i != j and on_right(res[i], res[j]) and
                (right[i] == -1 or res[right[i]].y > res[j].y))
                right[i] = j;
        if (right[i] != -1)
            has_left[right[i]] = true;
    }

    for (int i = 0; i < res_n; i++)
        if (res[i].val != -1 and !has_left[i]) {
            for (int j = i; j != -1 and res[j].val != -1; j = right[j]) {
                printf("%d", res[j].val);
                res[j].val = -1;
            }
            putchar('\n');
        }

    return 0;
}
}; // namespace predict