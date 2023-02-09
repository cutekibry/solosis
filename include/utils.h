#ifndef __UTILS_H__
#define __UTILS_H__

#include <chrono>

#define RMS_GAMMA 0.9f
#define RMS_EPS 1e-6f

#define GLOBAL_ETA 0.002f
#define REG_LAMBDA 0.001f

#define PREDICT_EPS 0.01f

#define TRAINING_NOISE 0.f

#define DROPOUT_P 0.3f

#define time_ms() std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

void create_parents_for_file(const char *file_path);
float sigma(float x);
float sigma_d(float x);

#define randcheck(p) (rand() <= RAND_MAX * (p))

float randf(float n);
float randarg(int in_size, int out_size);

#endif