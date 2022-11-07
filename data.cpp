#include "data.h"

int read_int(FILE *f) {
    int res;
    res = fgetc(f);
    res = res << 8 | fgetc(f);
    res = res << 8 | fgetc(f);
    res = res << 8 | fgetc(f);
    return res;
}

Dataset read_dataset(const char *label_file, const char *image_file, int &r, int &c)
{
    FILE *label_f, *image_f;
    int n;
    Dataset res;

    label_f = fopen(label_file, "rb");
    image_f = fopen(image_file, "rb");

    // read magic number
    read_int(label_f);
    read_int(image_f);
    // read n
    read_int(label_f);
    n = read_int(image_f);

    r = read_int(image_f);
    c = read_int(image_f);

    res.resize(n);
    for (int t = 0; t < n; t++)
    {
        res[t].input.resize(r * c);
        for (int i = 0; i < r * c; i++)
            res[t].input[i] = fgetc(image_f);
        res[t].output = fgetc(label_f);
    }

    fclose(label_f);
    fclose(image_f);
    return res;
}