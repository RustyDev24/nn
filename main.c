#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "matrix.h"

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

int main() {
  srand(time(NULL));

  double data[4][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
  };

  double rate = 0.1;

  matrix_t inputs = matrix_new(1, 2);

   // layer 1
  matrix_t weights_a = matrix_new(2, 2);
  matrix_t bias_a = matrix_new(1, 2);
  matrix_t delta_a = matrix_new(1, 2);
  matrix_t gradient_a = matrix_new(2, 2);
  matrix_fill_random(weights_a, -1, 1);
  matrix_fill_random(bias_a, -0.5, 0.5);
  
  // layer 2 (output)
  matrix_t weights_b = matrix_new(2, 1);
  matrix_t bias_b = matrix_new(1, 1);
  matrix_t delta_b = matrix_new(1, 1);
  matrix_t gradient_b = matrix_new(2, 1);
  matrix_fill_random(weights_b, -1, 1);
  matrix_fill_random(bias_b, -0.5, 0.5);

  matrix_t a1 = matrix_new(1, 2);
  matrix_t a2 = matrix_new(1, 1);

  for (int epoch = 0; epoch < 10000; epoch++) {
    for (int input = 0; input < 4; input++) {
      inputs.data[0] = data[input][0];
      inputs.data[1] = data[input][1];

      // layer 1
      matrix_dot(a1, inputs, weights_a);
      matrix_add(a1, a1, bias_a);
      matrix_apply(a1, sigmoid);

      // layer 2
      matrix_dot(a2, a1, weights_b);
      matrix_add(a2, a2, bias_b);
      matrix_apply(a2, sigmoid);

      /* ============================== */
      // backward pass
      delta_b.data[0] = (a2.data[0]- data[input][2]) * a2.data[0] * (1 - a2.data[0]);

      matrix_dot(delta_a, delta_b, matrix_transpose(weights_b));
      matrix_t derivatives = matrix_new(1, 2);
      for (int i = 0; i < a1.rows; i++) {
        for (int j = 0; j < a1.cols; j++) {
          derivatives.data[i*a1.cols + j] = a1.data[i*a1.cols + j] * (1 - a1.data[i*a1.cols + j]);
        }
      }
      matrix_hadamard(delta_a, delta_a, derivatives);

      matrix_dot(gradient_b, matrix_transpose(a1), delta_b);
      matrix_scale(gradient_b, rate);
      matrix_sub(weights_b, weights_b, gradient_b);
      matrix_scale(delta_b, rate);
      matrix_sub(bias_b, bias_b, delta_b);

      matrix_dot(gradient_a, matrix_transpose(inputs), delta_a);
      matrix_scale(gradient_a, rate);
      matrix_sub(weights_a, weights_a, gradient_a);
      matrix_scale(delta_a, rate);
      matrix_sub(bias_a, bias_a, delta_a);
    }
  }

  for (int input = 0; input < 4; input++) {
    inputs.data[0] = data[input][0];
    inputs.data[1] = data[input][1];

    // Layer 1
    matrix_dot(a1, inputs, weights_a);
    matrix_add(a1, a1, bias_a);
    matrix_apply(a1, sigmoid);

    // Layer 2
    matrix_dot(a2, a1, weights_b);
    matrix_add(a2, a2, bias_b);
    matrix_apply(a2, sigmoid);

    matrix_print(a2);
  }

  printf("FINAL\n");
  printf("weights\nA:\n");
  matrix_print(weights_a);
  printf("B:\n");
  matrix_print(weights_b);

  printf("bias\nA:\n");
  matrix_print(bias_a);
  printf("B:\n");
  matrix_print(bias_b);


 return 0;
}
