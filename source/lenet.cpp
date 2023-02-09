#include "lenet.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

Lenet::Lenet() {
    F7.test_mode = true;
    for (int i = 0; i < 10; i++)
        F7.keep[i] = true;
}
Lenet::Lenet(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (f == NULL) {
        fprintf(stderr, "Error(Lenet::Lenet(%s)): File %s open failed",
                filepath, filepath);
        exit(-3);
    }
    for (int i = 0; i < 6; i++)
        C1[i].read_from(f);
    for (int i = 0; i < 16; i++)
        C3[i].read_from(f);
    F5.read_from(f);
    F6.read_from(f);
    F7.read_from(f);
    F7.test_mode = true;
    for (int i = 0; i < 10; i++)
        F7.keep[i] = true;
    fclose(f);
}

void Lenet::forward_prop(float in[32][32]) {
    for (int i = 0; i < 6; i++) {
        C1[i].clear_out();
        C1[i].init_dropout();
        S2[i].clear_out();
    }
    for (int i = 0; i < 16; i++) {
        C3[i].clear_out();
        C3[i].init_dropout();
        S4[i].clear_out();
    }
    F5.clear_out();
    F5.init_dropout();
    F6.clear_out();
    F6.init_dropout();
    F7.clear_out();

    for (int i = 0; i < 6; i++)
        C1[i].forward_prop(in);

    for (int i = 0; i < 6; i++)
        S2[i].forward_prop(C1[i].out);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 6; j++)
            if (S2C3[i][j])
                C3[i].add_only(S2[j].out);
        C3[i].mod_out();
    }

    for (int i = 0; i < 16; i++)
        S4[i].forward_prop(C3[i].out);

    for (int i = 0, cnt = 0; i < 16; i++)
        for (int j = 0; j < 5; j++)
            for (int k = 0; k < 5; k++, cnt++)
                S4_out[cnt] = S4[i].out[j][k];
    F5.forward_prop(S4_out);
    F6.forward_prop(F5.out);
    F7.forward_prop(F6.out);
}

void Lenet::backward_prop(Data &data) {
    float expsum = 0;
    for (int i = 0; i < 10; i++) {
        F7.d_out[i] = expf(F7.out[i]);
        expsum += F7.d_out[i];
    }
    for (int i = 0; i < 10; i++)
        F7.d_out[i] /= expsum;
    F7.d_out[data.out]--;

    F7.backward_prop(F6.out, F6.d_out);

    F6.backward_prop(F5.out, F5.d_out);

    memset(S4_d_out, 0, sizeof(S4_d_out));
    F5.backward_prop(S4_out, S4_d_out);

    for (int i = 0, cnt = 0; i < 16; i++)
        for (int j = 0; j < 5; j++)
            for (int k = 0; k < 5; k++, cnt++)
                S4[i].d_out[j][k] = S4_d_out[cnt];
    for (int i = 0; i < 16; i++)
        S4[i].backward_prop(C3[i].out, C3[i].d_out);

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 6; j++)
            if (S2C3[i][j])
                C3[i].backward_prop(S2[j].out, S2[j].d_out);

    for (int i = 0; i < 6; i++)
        S2[i].backward_prop(C1[i].out, C1[i].d_out);

    for (int i = 0; i < 6; i++)
        C1[i].backward_prop(data.in, NULL);
}

void Lenet::train_batch(Dataset &dataset, int l, int n, float eta) {
    for (int i = 0; i < n; i++) {
        forward_prop(dataset[i + l].in);
        backward_prop(dataset[i + l]);
    }
    for (int i = 0; i < 6; i++)
        C1[i].add_delta(n, eta);
    for (int i = 0; i < 16; i++)
        C3[i].add_delta(n, eta);
    F5.add_delta(n, eta);
    F6.add_delta(n, eta);
    F7.add_delta(n, eta);
}

int Lenet::predict(float in[32][32]) {
    forward_prop(in);
    return std::max_element(F7.out, F7.out + 10) - F7.out;
}
float Lenet::error_rate(Dataset &dataset) {
    int cnt = 0;
    for (int i = 0; i < dataset.n; i++)
        if (predict(dataset[i].in) != dataset[i].out)
            cnt++;
    return (float)cnt / dataset.n;
}
void Lenet::error_and_loss(Dataset &dataset, float *error, float *loss) {
    *error = 0;
    *loss = 0;
    for (int i = 0; i < dataset.n; i++) {
        if (predict(dataset[i].in) != dataset[i].out)
            (*error)++;
        float expsum = 0;
        for (int j = 0; j < 10; j++)
            expsum += expf(F7.out[j]);
        *loss += logf(expsum) - F7.out[dataset[i].out];
    }
    *error /= dataset.n;
    *loss /= dataset.n;
}

void Lenet::save_file(const char *save_path, float train_err, float cross_err) {
    create_parents_for_file(save_path);
    FILE *f = fopen(save_path, "w");
    if (f == NULL) {
        fprintf(stderr, "Error(Lenet::save_file(%s)): File %s open failed",
                save_path, save_path);
        exit(-4);
    }
    for (int i = 0; i < 6; i++)
        C1[i].write_into(f);
    for (int i = 0; i < 16; i++)
        C3[i].write_into(f);
    F5.write_into(f);
    F6.write_into(f);
    F7.write_into(f);
    fprintf(f, "train: %.4f%%\n", train_err * 100);
    fprintf(f, "cross: %.4f%%\n", cross_err * 100);
    fclose(f);
}

void Lenet::toggle_test_mode(bool t) {
    for (int i = 0; i < 6; i++)
        C1[i].test_mode = t;
    for (int i = 0; i < 16; i++)
        C3[i].test_mode = t;
    F5.test_mode = t;
    F6.test_mode = t;
}