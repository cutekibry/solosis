#include "network.h"
#include "data.h"
#include "maths.h"

Network::Network() {}
Network::Network(float _eta, int _input_len) : eta(_eta), input_len(_input_len) {}

float Network::get_eta() const { return eta; }
void Network::set_eta(float _eta) { eta = _eta; }

void Network::forward_propagation(const dvector &input, std::vector<dvector> &x, std::vector<dvector> &y) const
{
  x.resize(b.size());
  y.resize(b.size());
  x[0] = input * W[0] + b[0];
  y[0] = sigmoid(x[0]);
  for (int i = 1; i < b.size(); i++)
  {
    x[i] = y[i - 1] * W[i] + b[i];
    y[i] = sigmoid(x[i]);
  }
}
dvector Network::forward_propagation(const dvector &input) const
{
  dvector x = input;
  for (int i = 0; i < b.size(); i++)
    x = sigmoid(x * W[i] + b[i]);
  return x;
}
void Network::back_propagation(const dvector &input, const int output, const std::vector<dvector> &x, const std::vector<dvector> &y, std::vector<dmatrix> &dW, std::vector<dvector> &db) const
{
  dvector dy;
  float expsum = 0;

  dy = y[b.size() - 1];
  for (int i = 0; i < dy.size(); i++)
    expsum += dy.get(i);
  expsum = exp(expsum);
  for (int i = 0; i < dy.size(); i++)
    dy[i] = exp(dy[i]) / expsum - (i == output);

  dW.resize(W.size());
  db.resize(b.size());

  for (int dep = b.size() - 1; dep >= 0; dep--)
  {
    dvector g = dy * sigmoid(y[dep]) * (1 - sigmoid(y[dep]));

    dW[dep] = combine(dep ? y[dep - 1] : input, g);
    db[dep] = g;
    dy = W[dep] * g;
  }
}

void Network::push_back(int n)
{
  W.push_back(rnd_dmatrix(b.empty() ? input_len : b.back().size(), n));
  b.push_back(rnd_dvector(n));
}
void Network::train(const Dataset &dataset)
{
  std::vector<dvector> x, y;
  std::vector<dmatrix> sumdW, dW;
  std::vector<dvector> sumdb, db;

  for (auto data : dataset)
  {
    forward_propagation(dvector(data.input), x, y);
    back_propagation(dvector(data.input), data.output, x, y, dW, db);
    if (sumdb.empty())
    {
      sumdW = dW;
      sumdb = db;
    }
    else
    {
      for (int i = 0; i < dW.size(); i++)
        sumdW[i] += dW[i];
      for (int i = 0; i < db.size(); i++)
        sumdb[i] += db[i];
    }
  }
  for (int i = 0; i < dW.size(); i++)
    W[i] -= sumdW[i] * (eta / dataset.size());
  for (int i = 0; i < db.size(); i++)
    b[i] -= sumdb[i] * (eta / dataset.size());
}

int Network::predict(const dvector &input) const { return forward_propagation(input).max_element_id(); }
float Network::loss(const Data &data) const
{
  dvector output = forward_propagation(dvector(data.input));
  float expsum = 0;
  for (int i = 0; i < output.size(); i++)
    expsum += exp(output[i]);
  return log(expsum) - output[data.output];
}
std::pair<float, float> Network::get_error_and_loss(const Dataset &dataset) const
{
  int cnt = 0;
  float sumloss = 0;
  for (auto data : dataset)
  {
    dvector output = forward_propagation(dvector(data.input));

    float expsum = 0;
    for (int i = 0; i < output.size(); i++)
      expsum += exp(output[i]);
    sumloss += log(expsum) - output[data.output];
    cnt += (output.max_element_id() != data.output);
  }
  return std::pair<float, float>((float)cnt / dataset.size(), sumloss / dataset.size());
}

void Network::write_file(const char *file_path) const
{
  FILE *file = fopen(file_path, "w");

  fprintf(file, "%d %f\n", input_len, eta);
  fprintf(file, "%lu\n", b.size());
  for (auto mat : W)
  {
    fprintf(file, "%d %d\n", mat.size().first, mat.size().second);
    for (int i = 0; i < mat.size().first; i++)
    {
      for (int j = 0; j < mat.size().second; j++)
        fprintf(file, "%f ", mat.get(i).get(j));
      fputc('\n', file);
    }
  }
  fprintf(file, "\n");
  for (auto vec: b)
  {
    fprintf(file, "%d\n", vec.size());
    for (int i = 0; i < vec.size(); i++)
      fprintf(file, "%f ", vec.get(i));
    fputc('\n', file);
  }
  fclose(file);
}

bool Network::read_file(const char *file_path)
{
  FILE *file = fopen(file_path, "r");

  if(file == NULL)
    return false;

  int n;

  fscanf(file, "%d %f\n", &input_len, &eta);
  fscanf(file, "%d\n", &n);

  b.resize(n);
  W.resize(n);

  for (auto &mat : W)
  {
    int x, y;
    fscanf(file, "%d %d\n", &x, &y);
    mat = dmatrix(x, y);
    for (int i = 0; i < mat.size().first; i++)
      for (int j = 0; j < mat.size().second; j++)
        fscanf(file, "%f", &mat[i][j]);
  }
  for (auto &vec : b)
  {
    int x;
    fscanf(file, "%d\n", &x);
    vec = dvector(x);
    for (int i = 0; i < vec.size(); i++)
      fscanf(file, "%f", &vec[i]);
  }

  fclose(file);
  return true;
}