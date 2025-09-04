#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "matrix.h"

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
  double rate = 0.1;

  matrix_t input = matrix_new(1, 2);

  // layer 1
  matrix_t weights_a = matrix_new(2, 2);
  matrix_t bias_a = matrix_new(1, 2);
  matrix_t delta_a = matrix_new(2, 1);
//  matrix_t gradient_a = matrix_new(2, 2);
  matrix_fill_random(weights_a, -1, 1);
  matrix_fill_random(bias_a, -0.5, 0.5);

//  delta_l = delta_l+1 * weights(l) (*) a'
  
  // layer 2 (output)
  matrix_t weights_b = matrix_new(2, 1);
  matrix_t bias_b = matrix_new(1, 1);
  matrix_t delta_b = matrix_new(1, 1);
//  matrix_t gradient_b = matrix_new(2, 1);
  matrix_fill_random(weights_b, -1, 1);
  matrix_fill_random(bias_b, -0.5, 0.5);

  matrix_t a1 = matrix_new(1, 2);
  matrix_t a2 = matrix_new(1, 1);

  for (int i = 0; i < 4; i++) {
    input.data[0] = data[i][0];
    input.data[1] = data[i][1];

    // Layer 1
    matrix_dot(a1, input, weights_a);
    matrix_add(a1, a1, bias_a);
    matrix_apply(a1, sigmoid);

    // Layer 2
    matrix_dot(a2, a1, weights_b);
    matrix_add(a2, a2, bias_b);
    matrix_apply(a2, sigmoid);

    /* for (int k = 0; k < a2.rows; k++) { */
    /*   for (int l = 0; l < a2.cols; l++) { */
    /*     printf("%f ", a2.data[k * a2.cols + l]); */
    /*   } */
    /*   printf("\n"); */
    /* } */

    /* ============================== */
    // Backward pass
    delta_b.data[0] = 2 * (input.data[2] - a2.data[0]) * a2.data[0] * (1 - a2.data[0]);

    matrix_dot(delta_a, delta_b, weights_b);
    matrix_t derivatives = matrix_new(1, 2);
    for (int i = 0; i < a1.rows; i++) {
      for (int j = 0; j < a1.cols; j++) {
        derivatives.data[i*a1.cols + j] = a1.data[i*a1.cols + j] * (1 - a1.data[i*a1.cols + j]);
      }
    }
    matrix_hadamard(delta_a, delta_a, derivatives);
    printf("==============================\n");
    matrix_print(delta_a);
    printf("==============================\n");
  }

  return 0;
}
