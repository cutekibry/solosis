#ifndef __DATA_H__
#define __DATA_H__

#include <cstdio>

class Data {
    public:
    Data();
    void normalize();
    void add_noise(float m);

    public:

    float in[32][32];
    int out;

};

class Dataset {
public:
    Dataset(const char *img_path, const char *lab_path);
    Dataset(Dataset &b, int l, int _n);
    ~Dataset();
    Data &operator[](int i);

    void add_noise(float m);

private:
    int read_int(FILE *f);

public:
    Data *datas;
    int n;
    int r, c;
};

#endif