#include "test.h"
#include "netset.h"

namespace test {

const int K = 10;

int main(int argc, char *argv[], void (*usage)(const char *prog)) {
    if (argc < 3) {
        usage(argv[0]);
        return -1;
    }

    
    Dataset train_set("mnist-data/train-images-idx3-ubyte",
                      "mnist-data/train-labels-idx1-ubyte");
    Dataset cross_set(train_set, 55000, 5000);
    Dataset test_set("mnist-data/t10k-images-idx3-ubyte",
                     "mnist-data/t10k-labels-idx1-ubyte");
    train_set.n = 55000;

    Netset netset(argc - 2, K);
    for (int i = 2; i < argc; i++)
        if (!netset.add_network(argv[i])) {
            fprintf(stderr, "Network %s add failed", argv[i]);
            return -2;
        }

    printf("train %.4f%%\n", netset.error_rate(train_set) * 100);
    printf("cross %.4f%%\n", netset.error_rate(cross_set) * 100);
    printf("test %.4f%%\n", netset.error_rate(test_set) * 100);
    return 0;
}
}; // namespace test