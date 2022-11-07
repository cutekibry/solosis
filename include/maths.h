#ifndef MATHS_H
#define MATHS_H

#include <vector>
#include <algorithm>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>

class dvector
{
private:
    std::vector<float> a;

public:
    dvector();
    dvector(const int n);
    dvector(const std::vector<unsigned char> &b);
    float &operator[](const int i);
    float get(const int i) const;
    void resize(const int n);
    int size() const;
    void reset();
    int max_element_id() const;


    dvector operator-() const;
    dvector operator+(const dvector &b) const;
    dvector operator-(const dvector &b) const;
    dvector operator*(const dvector &b) const;
    dvector operator+=(const dvector &b);
    dvector operator-=(const dvector &b);
    dvector operator*=(const dvector &b);

    dvector operator+(const float x) const;
    dvector operator-(const float x) const;
    dvector operator*(const float x) const;
    dvector operator/(const float x) const;
    dvector operator+=(const float x);
    dvector operator-=(const float x);
    dvector operator*=(const float x);
    dvector operator/=(const float x);

    friend dvector operator+(const float x, const dvector &b);
    friend dvector operator-(const float x, const dvector &b);
    friend dvector operator*(const float x, const dvector &b);
};

class dmatrix
{
private:
    std::vector<dvector> a;

public:
    dmatrix();
    dmatrix(int n, int m);
    dmatrix(std::pair<int, int>);
    dvector &operator[](const int i);
    dvector get(const int i) const;
    std::pair<int, int> size() const;

    void reset();

    dmatrix operator-() const;

    dmatrix operator+(const dmatrix &b) const;
    dmatrix operator-(const dmatrix &b) const;
    dmatrix operator+=(const dmatrix &b);
    dmatrix operator-=(const dmatrix &b);

    dmatrix operator*(const float x) const;
    dmatrix operator/(const float x) const;
    dmatrix operator*=(const float x);
    dmatrix operator/=(const float x);

    dvector to_dvector() const;
};

float sigmoid(const float x);
dvector sigmoid(const dvector &x);

float rnd();
dvector rnd_dvector(int n);
dmatrix rnd_dmatrix(int n, int m);

dmatrix combine(const dvector &a, const dvector &b);

dvector operator*(const dvector &a, const dmatrix &b);
dvector operator*(const dmatrix &a, const dvector &b);

#endif