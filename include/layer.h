#ifndef __LAYER_H__
#define __LAYER_H__

typedef enum { SIGMOID, RELU } ACTIVATION;

class Layer {
public:
    Layer(int _in_n, int _n, ACTIVATION _activate);
    ~Layer();

    void forward_propagation(float *in);
    void backward_propagation(float *prev_y, float *prev_dy, float eta);

private:
    inline float f(float x, ACTIVATION activate);
    inline float gradient(float x, ACTIVATION activate);

public:
    ACTIVATION activate;
    int in_n, n;
    float **W;
    float *b;
    float *y;
    float *dy;
};

#endif