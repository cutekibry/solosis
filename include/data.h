#ifndef __DATA_H__
#define __DATA_H__

#include <cstdio>

class Data {
    public:
    Data();
    Data(const char *img_path);
    void save_img(const char *save_path);

    public:

    float *in;
    int out;

};

class Dataset {
public:
    Dataset(const char *img_path, const char *lab_path);
    Dataset(Dataset &b, int l, int _n);
    ~Dataset();
    Data &operator[](int i);

private:
    int read_int(FILE *f);

public:
    Data *datas;
    int n;
    int r, c;
};

#endif