#include "network.h"
#include "data.h"
#include "maths.h"

Dataset train, test;

const float ETA_0 = 0.1;
const int T = 5;
const int STEP = 1;

int main()
{
  int r, c;

  Dataset train_set = read_dataset("data/train-labels-idx1-ubyte", "data/train-images-idx3-ubyte", r, c);
  Dataset test_set = read_dataset("data/t10k-labels-idx1-ubyte", "data/t10k-images-idx3-ubyte", r, c);
  assert(r == c);

  int _L = 0;
  int _R = 1000;
  train_set = Dataset(train_set.begin() + _L, train_set.begin() + _R);
  test_set.resize(100);

  Network network;
  
  if(!network.read_file("network-data")) {
    network = Network(ETA_0, r * c);
    network.push_back(sqrt(r * c));
    network.push_back(10);
  }

  printf("# train_n = %lu\n", train_set.size());
  printf("# test_n = %lu\n", test_set.size());
  printf("# r = c = %d\n", r);
  printf("# eta = %g\n", network.get_eta());

  std::pair<float, float> pp;

  printf("# trained %3d time(s):         error        loss\n", 0);
  pp = network.get_error_and_loss(train_set);
  printf("#                      train   %-6.2f%%      %.4f\n", pp.first * 100, pp.second);
  pp = network.get_error_and_loss(test_set);
  printf("#                      test    %-6.2f%%      %.4f\n", pp.first * 100, pp.second);
  for (int t = 1; t <= T; t++)
  {
    network.train(train_set);
    if (t % STEP == 0)
    {
      // printf("# trained %3d time(s): eta    %5.2f  temp %.3f\n", t, network.get_eta(), temp);
      printf("# trained %3d time(s):         error        loss\n", t);
      pp = network.get_error_and_loss(train_set);
      printf("#                      train   %-6.2f%%      %.4f\n", pp.first * 100, pp.second);
      pp = network.get_error_and_loss(test_set);
      printf("#                      test    %-6.2f%%      %.4f\n", pp.first * 100, pp.second);
      putchar('\n');
    }
  }
  network.write_file("network-data");
  return 0;
}