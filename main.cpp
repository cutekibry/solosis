#include "predict.h"
#include "train.h"
#include "test.h"
#include <cstdio>
#include <cstring>

void usage(const char *prog) {
    printf("usage: %s train [load_model_path] [save_model_path] [train_times] "
           "[eta_max] [eta_min]\n",
           prog);
    printf("       %s train --new [save_model_path] [train_times] [eta_max] "
           "[eta_min] [srand_seed] [in_n] [layer1_n] ... [layerk_n] [out_n]\n",
           prog);
    printf("\n");
    printf("       %s test [network1_path] ... [networkk_path]\n", prog);
    printf("\n");
    printf("       %s predict [image_path]\n", prog);
    printf("       %s predict [image_path] --debug\n", prog);
}

int main(int argc, char *argv[]) {
    int r, c;

    if (argc == 1) {
        usage(argv[0]);
        return -1;
    }
    if (strcmp(argv[1], "train") == 0)
        return train::main(argc, argv, usage);
    else if (strcmp(argv[1], "predict") == 0)
        return predict::main(argc, argv, usage);
    else if (strcmp(argv[1], "test") == 0) 
        return test::main(argc, argv, usage);

    usage(argv[0]);
    return -1;
}