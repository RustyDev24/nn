#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"

static double rand_double() {
  return (double)rand() / (double) RAND_MAX;
}

matrix_t matrix_new(size_t rows, size_t cols) {
  matrix_t matrix = {0};
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.data = (double *) calloc (rows * cols, sizeof(double));

  return matrix;
}

void matrix_add(matrix_t c, matrix_t a, matrix_t b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      c.data[i*c.cols + j] = a.data[i*a.cols + j] + b.data[i*b.cols + j];
    }
  }
}

void matrix_sub(matrix_t c, matrix_t a, matrix_t b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      c.data[i*c.cols + j] = a.data[i*a.cols + j] - b.data[i*b.cols + j];
    }
  }
}

void matrix_dot(matrix_t c, matrix_t a, matrix_t b) {
  assert(a.cols == b.rows);
  assert(c.cols == b.cols);
  assert(c.rows == a.rows);
  for (int i = 0; i < a.rows; i++) {
    for (int k = 0; k < b.cols; k++) {
      for (int j = 0; j < a.cols; j++) {
        c.data[i*b.cols + k] += a.data[i*a.cols + j] * b.data[j*b.cols + k];
      }
    }
  }
}

void matrix_hadamard(matrix_t c, matrix_t a, matrix_t b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      c.data[i*a.cols + j] = a.data[i*a.cols + j] * b.data[i*a.cols + j];
    }
  }
}

void matrix_fill_random(matrix_t mat, int low, int high) {
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      mat.data[i*mat.cols + j] = rand_double() * (high  - low) + low;
    }
  }
}

matrix_t matrix_transpose(matrix_t mat) {
  matrix_t transpose = matrix_new(mat.cols, mat.rows);
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      transpose.data[j*transpose.cols + i] = mat.data[i*mat.cols + j];
    }
  }

  return transpose;
}

void matrix_scale(matrix_t mat, double scaler) {
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      mat.data[i*mat.cols + j] *= scaler;
    }
  }
}

void matrix_print(matrix_t mat) {
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      printf("%f ", mat.data[i*mat.cols + j]);
    }
    printf("\n");
  }
}

void matrix_apply(matrix_t mat, double (*func) (double)) {
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      mat.data[i*mat.cols + j] = func(mat.data[i*mat.cols + j]);
    }
  }
}
