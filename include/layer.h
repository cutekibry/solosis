#ifndef __LAYER_H__
#define __LAYER_H__

#include <cstdio>

template <int in_size, int kernal_size> class Conv_layer {
    static const int out_size = in_size - kernal_size + 1;

public:
    Conv_layer();
    void init_dropout();

    void forward_prop(float in[in_size][in_size]);
    void backward_prop(float in[in_size][in_size],
                       float d_in[in_size][in_size]);
    void add_only(float in[in_size][in_size]);
    void mod_out();
    void clear_out();
    void add_delta(int data_n, float eta);

    void read_from(FILE *f);
    void write_into(FILE *f);

public:
    bool test_mode;
    float w[kernal_size][kernal_size];
    float d_w[kernal_size][kernal_size];
    float r_w[kernal_size][kernal_size];
    bool keep[out_size][out_size];
    float out[out_size][out_size];
    float d_out[out_size][out_size];
};

template <int in_size, int kernal_size> class Subsamp_layer {
    static const int out_size = in_size / kernal_size;

public:
    Subsamp_layer();
    void forward_prop(float in[in_size][in_size]);
    void backward_prop(float in[in_size][in_size],
                       float d_in[in_size][in_size]);
    void add_only(float in[in_size][in_size]);
    void mod_out();
    void clear_out();

public:
    float out[out_size][out_size];
    float d_out[out_size][out_size];
};

template <int in_size, int out_size> class Dense_layer {
public:
    Dense_layer();

    void init_dropout();

    void forward_prop(float in[in_size]);
    void backward_prop(float in[in_size], float d_in[in_size]);
    void clear_out();
    void add_delta(int data_n, float eta);

    void read_from(FILE *f);
    void write_into(FILE *f);

public:
    bool test_mode;
    bool keep[out_size];
    float w[out_size][in_size];
    float d_w[out_size][in_size];
    float r_w[out_size][in_size];
    float b[out_size];
    float d_b[out_size];
    float r_b[out_size];
    float out[out_size];
    float d_out[out_size];
};

#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

// Class Conv_layer
template <int in_size, int kernal_size>
Conv_layer<in_size, kernal_size>::Conv_layer() : test_mode(false) {
    clear_out();
    memset(d_w, 0, sizeof(d_w));
    memset(r_w, 0, sizeof(r_w));
    for (int i = 0; i < kernal_size; i++)
        for (int j = 0; j < kernal_size; j++)
            w[i][j] = randarg(in_size * in_size, out_size * out_size);
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::init_dropout() {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < out_size; j++)
            keep[i][j] = !randcheck(DROPOUT_P);
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::forward_prop(
    float in[in_size][in_size]) {
    add_only(in);
    mod_out();
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::backward_prop(
    float in[in_size][in_size], float d_in[in_size][in_size]) {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < out_size; j++)
            if (keep[i][j]) {
                float g = d_out[i][j] * sigma_d(out[i][j]);
                for (int x = 0; x < kernal_size; x++)
                    for (int y = 0; y < kernal_size; y++) {
                        d_w[x][y] += g * in[i + x][j + y];
                        if (d_in != NULL)
                            d_in[i + x][j + y] += g * w[x][y];
                    }
            }
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::add_only(float in[in_size][in_size]) {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < out_size; j++)
            if (test_mode or keep[i][j]) {
                float res = 0;
                for (int x = 0; x < kernal_size; x++)
                    for (int y = 0; y < kernal_size; y++)
                        res += w[x][y] * in[i + x][j + y];
                out[i][j] = res;
            }
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::mod_out() {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < out_size; j++)
            out[i][j] = sigma(out[i][j]);
    if (test_mode)
        for (int i = 0; i < out_size; i++)
            for (int j = 0; j < out_size; j++)
                out[i][j] *= 1 - DROPOUT_P;
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::clear_out() {
    memset(out, 0, sizeof(out));
    memset(d_out, 0, sizeof(d_out));
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::add_delta(int data_n, float eta) {
    for (int x = 0; x < kernal_size; x++)
        for (int y = 0; y < kernal_size; y++) {
            d_w[x][y] /= data_n;
            r_w[x][y] =
                RMS_GAMMA * r_w[x][y] + (1 - RMS_GAMMA) * d_w[x][y] * d_w[x][y];
            w[x][y] -= eta / sqrtf(r_w[x][y] + RMS_EPS) *
                       (d_w[x][y] + REG_LAMBDA * w[x][y]);
            d_w[x][y] = 0;
        }
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::read_from(FILE *f) {
    for (int x = 0; x < kernal_size; x++)
        for (int y = 0; y < kernal_size; y++)
            fscanf(f, "%f", &w[x][y]);
    for (int x = 0; x < kernal_size; x++)
        for (int y = 0; y < kernal_size; y++)
            fscanf(f, "%f", &r_w[x][y]);
}
template <int in_size, int kernal_size>
void Conv_layer<in_size, kernal_size>::write_into(FILE *f) {
    for (int x = 0; x < kernal_size; x++)
        for (int y = 0; y < kernal_size; y++)
            fprintf(f, "%g ", w[x][y]);
    for (int x = 0; x < kernal_size; x++)
        for (int y = 0; y < kernal_size; y++)
            fprintf(f, "%g ", r_w[x][y]);
    fprintf(f, "\n");
}

// Class Subsamp_layer
template <int in_size, int kernal_size>
Subsamp_layer<in_size, kernal_size>::Subsamp_layer() {
    static_assert(in_size % kernal_size == 0);
    clear_out();
}

template <int in_size, int kernal_size>
void Subsamp_layer<in_size, kernal_size>::forward_prop(
    float in[in_size][in_size]) {
    add_only(in);
    mod_out();
}
template <int in_size, int kernal_size>
void Subsamp_layer<in_size, kernal_size>::backward_prop(
    float in[in_size][in_size], float d_in[in_size][in_size]) {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < out_size; j++)
            for (int x = 0; x < kernal_size; x++)
                for (int y = 0; y < kernal_size; y++)
                    d_in[i * kernal_size + x][j * kernal_size + y] +=
                        d_out[i][j] / (kernal_size * kernal_size);
}
template <int in_size, int kernal_size>
void Subsamp_layer<in_size, kernal_size>::add_only(float in[in_size][in_size]) {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < out_size; j++)
            for (int x = 0; x < kernal_size; x++)
                for (int y = 0; y < kernal_size; y++)
                    out[i][j] += in[i * kernal_size + x][j * kernal_size + y];
}
template <int in_size, int kernal_size>
void Subsamp_layer<in_size, kernal_size>::mod_out() {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < out_size; j++)
            out[i][j] /= kernal_size * kernal_size;
}
template <int in_size, int kernal_size>
void Subsamp_layer<in_size, kernal_size>::clear_out() {
    memset(out, 0, sizeof(out));
    memset(d_out, 0, sizeof(d_out));
}

// class Dense_layer
template <int in_size, int out_size>
Dense_layer<in_size, out_size>::Dense_layer() : test_mode(false) {
    clear_out();
    memset(d_w, 0, sizeof(d_w));
    memset(r_w, 0, sizeof(r_w));
    memset(d_b, 0, sizeof(d_b));
    memset(r_b, 0, sizeof(r_b));
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++)
            w[i][j] = randarg(in_size + 1, out_size);
        b[i] = randarg(in_size + 1, out_size);
    }
}
template <int in_size, int out_size>
void Dense_layer<in_size, out_size>::init_dropout() {
    for (int i = 0; i < out_size; i++)
        keep[i] = !randcheck(DROPOUT_P);
}

template <int in_size, int out_size>
void Dense_layer<in_size, out_size>::forward_prop(float in[in_size]) {
    for (int i = 0; i < out_size; i++)
        if (test_mode or keep[i]) {
            float res = 0;
            for (int j = 0; j < in_size; j++)
                res += w[i][j] * in[j];
            out[i] = sigma(res);
        }

    if (test_mode)
        for (int i = 0; i < out_size; i++)
            out[i] *= 1 - DROPOUT_P;
}
template <int in_size, int out_size>
void Dense_layer<in_size, out_size>::backward_prop(float in[in_size],
                                                   float d_in[in_size]) {
    for (int i = 0; i < out_size; i++)
        if (keep[i]) {
            float g = d_out[i] * sigma_d(out[i]);
            for (int j = 0; j < in_size; j++) {
                d_w[i][j] += in[j] * g;
                d_in[j] += w[i][j] * g;
            }
            d_b[i] += g;
        }
}
template <int in_size, int out_size>
void Dense_layer<in_size, out_size>::clear_out() {
    memset(out, 0, sizeof(out));
    memset(d_out, 0, sizeof(d_out));
}
template <int in_size, int out_size>
void Dense_layer<in_size, out_size>::add_delta(int data_n, float eta) {
    float inv_data_n = 1.f / data_n;
    for (int i = 0; i < out_size; i++)
        if (keep[i]) {
            for (int j = 0; j < in_size; j++) {
                float t = d_w[i][j] * inv_data_n;
                r_w[i][j] = RMS_GAMMA * r_w[i][j] + (1 - RMS_GAMMA) * t * t;
                w[i][j] -= eta * (t + REG_LAMBDA * w[i][j]) /
                           sqrtf(r_w[i][j] + RMS_EPS);
                d_w[i][j] = 0;
            }
            float t = d_b[i] * inv_data_n;
            r_b[i] = RMS_GAMMA * r_b[i] + (1 - RMS_GAMMA) * t * t;
            b[i] -= eta * (t + REG_LAMBDA * b[i]) / sqrtf(r_b[i] + RMS_EPS);
            d_b[i] = 0;
        }
}
template <int in_size, int out_size>
void Dense_layer<in_size, out_size>::read_from(FILE *f) {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < in_size; j++)
            fscanf(f, "%f", &w[i][j]);
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < in_size; j++)
            fscanf(f, "%f", &r_w[i][j]);
    for (int i = 0; i < out_size; i++)
        fscanf(f, "%f", &b[i]);
    for (int i = 0; i < out_size; i++)
        fscanf(f, "%f", &r_b[i]);
}
template <int in_size, int out_size>
void Dense_layer<in_size, out_size>::write_into(FILE *f) {
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < in_size; j++)
            fprintf(f, "%g ", w[i][j]);
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < in_size; j++)
            fprintf(f, "%g ", r_w[i][j]);
    for (int i = 0; i < out_size; i++)
        fprintf(f, "%g ", b[i]);
    for (int i = 0; i < out_size; i++)
        fprintf(f, "%g ", r_b[i]);
    fprintf(f, "\n");
}

#endif
