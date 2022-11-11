#ifndef __NETSET_H__
#define __NETSET_H__

#include "network.h"

class Netset {
public:
    Netset(int _cap, int _out_n);
    ~Netset();

    bool add_network(const char *file);

    void predict_p(float *in);
    int predict(float *in);

    float error_rate(Dataset &dataset);

public:
    int n;
    int cap, out_n;
    Network **networks;
    float *out;
};

#endif