#include <stdio.h>
#include <stdint.h>
#include <math.h>

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

int main() {
  double data[4][3] = {
    {0, 0, 1},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
  };

  double weight_x = 1;
  double weight_y = 1;
  double rate = 0.5;
  double bias = -1;

  for (int i = 0; i < 1000; i++) {
    double error = 0;
    for (int j = 0; j < 4; j++) {
      double output = sigmoid((data[j][0] * weight_x + data[j][1] * weight_y) + bias);
      error += pow(data[j][2] - output, 2);

      double delta = ((output - data[j][2]) * 2) * output * (1 - output);
      double grad_x = delta * data[j][0];
      double grad_y = delta * data[j][1];
      double grad_b = delta;

      weight_x -= rate * grad_x;
      weight_y -= rate * grad_y;
      bias -= rate * grad_b;
    }

    if (error / 4 < 0.0001) {
      printf("model trained\n");
      break;
    }

    if (i % 10 == 0) {
      printf("%d iterations complete\n", i);
      printf("updated weights: weight_x = %f, weight_y = %f\n", weight_x, weight_y);
      printf("updated bias: bias = %f\n", bias);
      printf("total error: %f\n", error/4);
    }
  }

  printf("Final values\n");
  printf("Weights: w_x: %f\tw_y: %f\n", weight_x, weight_y);
  printf("Bias: %f\n", bias);

  for (int i = 0; i < 4; i++) {
    double output = sigmoid((data[i][0] * weight_x + data[i][1] * weight_y) + bias);
    printf("%f AND %f = %f (expected: %f)\n", data[i][0], data[i][1], output, data[i][2]);
  }
 
  return 0;
}
