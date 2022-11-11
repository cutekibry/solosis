#include "train.h"
#include "data.h"
#include "layer.h"
#include "network.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>

namespace train {

void random_shuffle(Data *a, int n) {
    for (int i = 1; i < n; i++)
        std::swap(a[rand() % i], a[i]);
}

int main(int argc, char *argv[], void (*usage)(const char *prog)) {
    float eta_max, eta_min;
    char *load_model_path, *save_model_path;
    int train_times;
    Network *network;

    if (argc < 7) {
        usage(argv[0]);
        return -1;
    }

    Dataset train_set("mnist-data/train-images-idx3-ubyte",
                      "mnist-data/train-labels-idx1-ubyte");
    Dataset cross_set(train_set, 55000, 5000);
    Dataset test_set("mnist-data/t10k-images-idx3-ubyte",
                     "mnist-data/t10k-labels-idx1-ubyte");
                     train_set.n = 55000;
    float train_error, test_error, cross_error;
    float train_loss, test_loss, cross_loss;

    if (strcmp(argv[2], "--new") == 0) {
        if (argc < 10) {
            usage(argv[0]);
            return -1;
        }
        save_model_path = argv[3];
        train_times = atoi(argv[4]);
        eta_max = atof(argv[5]);
        eta_min = atof(argv[6]);

        srand(atoi(argv[7]));
        network = new Network(atoi(argv[8]), argc - 9, SOFTMAX, eta_max);
        for (int i = 9; i < argc; i++)
            network->add_layer(atoi(argv[i]), SIGMOID);

    } else {
        if (argc < 7) {
            usage(argv[0]);
            return -1;
        }
        load_model_path = argv[2];
        save_model_path = argv[3];
        train_times = atoi(argv[4]);
        eta_max = atof(argv[5]);
        eta_min = atof(argv[6]);

        network = new Network(load_model_path);
        if (network->layer_n == -1) {
            printf("Read model failed\n");
            return -2;
        }
        network->eta = eta_max;
    }

    assert(train_set.r * train_set.c == network->in_n);

    for (int t = 1; t <= train_times; t++) {
        random_shuffle(train_set.datas, train_set.n);
        for (int i = 0; i < train_set.n; i++) {
            network->train(train_set[i], network->eta);
            network->eta -= (eta_max - eta_min) / (train_set.n * train_times);
            if ((i + 1) / (train_set.n / 10) > i / (train_set.n / 10)) {
                printf(">");
                fflush(stdout);
            }
        }
        printf("\n");
        char dpath[500];
        sprintf(dpath, "%s/ver%d", save_model_path, t);

        printf("# %d finished\n", t);

        printf("#          error      loss\n");
        network->get_error_and_loss(train_set, &train_error, &train_loss);
        printf("# train    %-7.3f%%     %.6f\n", train_error * 100, train_loss);
        network->get_error_and_loss(cross_set, &cross_error, &cross_loss);
        printf("# cross    %-7.3f%%     %.6f\n", cross_error * 100, cross_loss);
        network->get_error_and_loss(test_set, &test_error, &test_loss);
        printf("# test     %-7.3f%%     %.6f\n", test_error * 100, test_loss);
        putchar('\n');

        network->write_file(dpath, train_error, cross_error, test_error);
    }

    return 0;
}
}; // namespace train