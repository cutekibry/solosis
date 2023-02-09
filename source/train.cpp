#include "train.h"
#include "data.h"
#include "layer.h"
#include "lenet.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace train {

int t;
float *train_err_a, *train_los_a, *cross_err_a, *cross_los_a;

void random_shuffle(Data *a, int n) {
    for (int i = 1; i < n; i++)
        std::swap(a[rand() % i], a[i]);
}
void print_array(const char *array_name, float *a, int n) {
    if (n == 0) {
        printf("%s = []\n", array_name);
        return;
    }

    printf("%s = [%g", array_name, a[0]);
    for (int i = 1; i < n; i++)
        printf(", %g", a[i]);
    printf("]\n");
}

void signal_handler(int signum) {
    printf("Interrupt signal (%d) received.\n", signum);
    printf("# Current versions count: %d\n", t);
    print_array("train_err", train_err_a, t);
    print_array("cross_err", cross_err_a, t);
    print_array("train_los", train_los_a, t);
    print_array("cross_los", cross_los_a, t);
    exit(signum);
}

int main(int argc, char *argv[], void (*usage)(const char *prog)) {
    char *load_model_path, *save_model_path;
    float eta;
    int train_times, batch_size;
    Lenet *lenet;

    signal(SIGINT, signal_handler);

    if (argc < 7) {
        usage(argv[0]);
        return -1;
    }

    Dataset train_set("mnist-data/train-images-idx3-ubyte",
                      "mnist-data/train-labels-idx1-ubyte");

    train_set.add_noise(TRAINING_NOISE);

    Dataset cross_set(train_set, 55000, 5000);
    train_set.n = 55000;

    load_model_path = argv[2];
    save_model_path = argv[3];
    train_times = atoi(argv[4]);
    batch_size = atoi(argv[5]);
    eta = atof(argv[6]) * GLOBAL_ETA;

    train_err_a = new float[train_times];
    cross_err_a = new float[train_times];
    train_los_a = new float[train_times];
    cross_los_a = new float[train_times];

    if (strcmp(load_model_path, "--new") == 0)
        lenet = new Lenet();
    else
        lenet = new Lenet(load_model_path);

    for (t = 0; t < train_times; t++) {
        random_shuffle(train_set.datas, train_set.n);

        long long cur_ms = time_ms();

        int last_progress = 0;
        for (int l = 0; l < train_set.n; l += batch_size) {
            lenet->train_batch(train_set, l,
                               std::min(batch_size, train_set.n - l), eta);
            int progress =
                std::min(l + batch_size, train_set.n) * 100 / train_set.n + 0.5;
            if (progress > last_progress) {
                for (int i = 0; i < 15; i++)
                    putchar(0x08);
                printf("[%d%%]", progress);
                fflush(stdout);
            }
            last_progress = progress;
        }
        for (int i = 0; i < 15; i++)
            putchar(0x08);
        char dpath[strlen(save_model_path) + 15];
        sprintf(dpath, "%s/ver%d", save_model_path, t + 1);

        printf("# %d finished in %.3fs, with eta = %g\n", t + 1, (time_ms() - cur_ms) / 1000.0f, eta);

        float train_err, cross_err, train_los, cross_los;

        lenet->toggle_test_mode(true);
        lenet->error_and_loss(train_set, &train_err, &train_los);
        lenet->error_and_loss(cross_set, &cross_err, &cross_los);
        lenet->toggle_test_mode(false);
        printf("#          error            loss\n");
        printf("# train    %-9.4f%%    %.6f\n", train_err * 100, train_los);
        printf("# cross    %-9.4f%%    %.6f\n", cross_err * 100, cross_los);
        lenet->save_file(dpath, train_err, cross_err);

        train_err_a[t] = train_err;
        cross_err_a[t] = cross_err;
        train_los_a[t] = train_los;
        cross_los_a[t] = cross_los;

        eta *= 1 - 0.03f;
    }

    print_array("train_err", train_err_a, train_times);
    print_array("cross_err", cross_err_a, train_times);
    print_array("train_los", train_los_a, train_times);
    print_array("cross_los", cross_los_a, train_times);

    return 0;
}
}; // namespace train