#include "maths.h"

// dvector
dvector::dvector() {}
dvector::dvector(const int n) { a.resize(n); }
dvector::dvector(const std::vector<unsigned char> &b)
{
  a.resize(b.size());
  for (int i = 0; i < a.size(); i++)
    a[i] = b[i];
}
float &dvector::operator[](const int i) { return a[i]; }
float dvector::get(const int i) const { return a[i]; }
void dvector::resize(const int n) { a.resize(n); }
int dvector::size() const { return a.size(); }
void dvector::reset()
{
  for (int i = 0; i < a.size(); i++)
    a[i] = 0;
}
int dvector::max_element_id() const { return std::max_element(a.begin(), a.end()) - a.begin(); }

dvector dvector::operator-() const
{
  dvector res = *this;
  for (int i = 0; i < size(); i++)
    res[i] = -res[i];
  return res;
}

dvector dvector::operator+(const dvector &b) const
{
  dvector res = *this;
  assert(size() == b.size());
  for (int i = 0; i < size(); i++)
    res[i] += b.get(i);
  return res;
}
dvector dvector::operator-(const dvector &b) const { return (*this) + (-b); }
dvector dvector::operator*(const dvector &b) const
{
  dvector res = *this;
  assert(size() == b.size());
  for (int i = 0; i < size(); i++)
    res[i] *= b.get(i);
  return res;
}
dvector dvector::operator+=(const dvector &b) { return (*this) = (*this) + b; }
dvector dvector::operator-=(const dvector &b) { return (*this) = (*this) - b; }
dvector dvector::operator*=(const dvector &b) { return (*this) = (*this) * b; }

dvector dvector::operator+(const float x) const
{
  dvector res = *this;
  for (int i = 0; i < size(); i++)
    res[i] += x;
  return res;
}
dvector dvector::operator-(const float x) const { return (*this) + (-x); }
dvector dvector::operator*(const float x) const
{
  dvector res = *this;
  for (int i = 0; i < size(); i++)
    res[i] *= x;
  return res;
}
dvector dvector::operator/(const float x) const { return (*this) * (1 / x); }
dvector dvector::operator+=(const float x) { return (*this) = (*this) + x; }
dvector dvector::operator-=(const float x) { return (*this) = (*this) - x; }
dvector dvector::operator*=(const float x) { return (*this) = (*this) * x; }
dvector dvector::operator/=(const float x) { return (*this) = (*this) / x; }

dvector operator+(const float x, const dvector &b) { return b + x; }
dvector operator-(const float x, const dvector &b) { return (-b) + x; }
dvector operator*(const float x, const dvector &b) { return b * x; }

// dmatrix
dmatrix::dmatrix() {}
dmatrix::dmatrix(int n, int m)
{
  a.resize(n);
  for (int i = 0; i < n; i++)
    a[i].resize(m);
}
dmatrix::dmatrix(std::pair<int, int> p) { (*this) = dmatrix(p.first, p.second); }
std::pair<int, int> dmatrix::size() const { return std::pair<int, int>(a.size(), a[0].size()); }
dvector &dmatrix::operator[](const int i) { return a[i]; }
dvector dmatrix::get(const int i) const { return a[i]; }
void dmatrix::reset()
{
  for (int i = 0; i < a.size(); i++)
    a[i].reset();
}

dmatrix dmatrix::operator-() const
{
  dmatrix res = *this;
  for (int i = 0; i < a.size(); i++)
    res[i] = -res[i];
  return res;
}

dmatrix dmatrix::operator+(const dmatrix &b) const
{
  dmatrix res = *this;
  assert(a.size() == b.a.size());
  for (int i = 0; i < a.size(); i++)
    res[i] += b.get(i);
  return res;
}
dmatrix dmatrix::operator-(const dmatrix &b) const { return (*this) + (-b); }
dmatrix dmatrix::operator+=(const dmatrix &b) { return (*this) = (*this) + b; }
dmatrix dmatrix::operator-=(const dmatrix &b) { return (*this) = (*this) - b; }

dmatrix dmatrix::operator*(const float x) const
{
  dmatrix res = *this;
  for (int i = 0; i < a.size(); i++)
    res[i] *= x;
  return res;
}
dmatrix dmatrix::operator/(const float x) const { return (*this) * (1 / x); }
dmatrix dmatrix::operator*=(const float x) { return (*this) = (*this) * x; }
dmatrix dmatrix::operator/=(const float x) { return (*this) = (*this) / x; }

dvector dmatrix::to_dvector() const
{
  dvector res(a.size() * a[0].size());
  int k = 0;
  for (int i = 0; i < a.size(); i++)
    for (int j = 0; j < a[i].size(); j++)
      res[k++] = a[i].get(j);
  return res;
}

dvector operator*(const dvector &a, const dmatrix &b)
{
  dvector c(b.size().second);

  assert(a.size() == b.size().first);

  for (int i = 0; i < a.size(); i++)
    for (int j = 0; j < c.size(); j++)
      c[j] += a.get(i) * b.get(i).get(j);
  return c;
}
dvector operator*(const dmatrix &a, const dvector &b)
{
  dvector c(a.size().first);

  assert(a.size().second == b.size());

  for (int i = 0; i < c.size(); i++)
    for (int j = 0; j < b.size(); j++)
      c[i] += a.get(i).get(j) * b.get(j);
  return c;
}

// sigmoid
float sigmoid(const float x) { return 1 / (1 + exp(-x)); }
dvector sigmoid(const dvector &x)
{
  dvector y = x;
  for (int i = 0; i < y.size(); i++)
    y[i] = sigmoid(y[i]);
  return y;
}

// random
float rnd() { return (float)rand() / RAND_MAX; }
dvector rnd_dvector(int n)
{
  dvector res(n);
  for (int i = 0; i < n; i++)
    res[i] = rnd() - 0.5;
  return res;
}
dmatrix rnd_dmatrix(int n, int m)
{
  dmatrix res(n, m);
  for (int i = 0; i < n; i++)
    res[i] = rnd_dvector(m);
  return res;
}

dmatrix combine(const dvector &a, const dvector &b)
{
  dmatrix res(a.size(), b.size());
  for (int i = 0; i < a.size(); i++)
    for (int j = 0; j < b.size(); j++)
      res[i][j] = a.get(i) * b.get(j);
  return res;
}

dvector onehot(int n, int x)
{
  dvector res(n);
  res[x] = 1;
  return res;
}