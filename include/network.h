#ifndef NETWORK_H
#define NETWORK_H

#include "maths.h"
#include "data.h"

/*
layer[-1] = input
layer[i] = sigmoid(layer[i - 1] * W[i] + b[i])
*/

class Network
{
private:
    std::vector<dmatrix> W;
    std::vector<dvector> b;
    float eta;
    int input_len;

    void forward_propagation(const dvector &input, std::vector<dvector> &x, std::vector<dvector> &y) const;
    dvector forward_propagation(const dvector &input) const;

    void back_propagation(const dvector &input, const int output, const std::vector<dvector> &x, const std::vector<dvector> &y, std::vector<dmatrix> &dW, std::vector<dvector> &db) const;

public:
    Network();
    Network(float _eta, int _input_len);

    float get_eta() const;
    void set_eta(float _eta);

    void push_back(int n);
    void train(const Dataset &dataset);

    int predict(const dvector &input) const;
    float loss(const Data &data) const;
    std::pair<float, float> get_error_and_loss(const Dataset &dataset) const;

    void write_file(const char *file_path) const;
    bool read_file(const char *file_path);
};
#endif