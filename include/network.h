#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "data.h"
#include "layer.h"

typedef enum { SOFTMAX } LOSSTYPE;

class Network {
public:
    Network();
    Network(int _in_n, int _max_layer_n, LOSSTYPE _losstype, float _eta);
    Network(const char *file);
    ~Network();

    void add_layer(int n, ACTIVATION activation);

    void forward_propagation(float *in);
    void back_propagation(float *in, int out, float eta);
    void train(Data data, float eta);

    float *get_out();
    Layer *get_last_layer();

    int predict(float *in);
    void predict_p(float *in);
    void get_error_and_loss(Dataset &dataset, float *error, float *ls);

    void write_file(const char *file, float train_error, float cross_error,
                    float test_error);

private:
    void calc_dy(Layer *layer, int out);
    float calc_loss(Layer *layer, int out);

public:
    LOSSTYPE losstype;
    int in_n;
    int layer_n;
    int max_layer_n;
    float eta;
    Layer **layers;
};

#endif