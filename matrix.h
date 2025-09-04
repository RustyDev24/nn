#ifndef ML_MATH_H
#define ML_MATH_H

typedef struct {
  size_t rows;
  size_t cols;
  double *data;
} matrix_t;

matrix_t matrix_new(size_t rows, size_t cols);
void matrix_add(matrix_t c, matrix_t a, matrix_t b);
void matrix_dot(matrix_t c, matrix_t a, matrix_t b);
void matrix_hadamard(matrix_t c, matrix_t a, matrix_t b);
void matrix_fill_random(matrix_t mat, int low, int high);
void matrix_print(matrix_t mat);

// Apply a function onto all the elements of the matrix (map over matrix elements)
void matrix_apply(matrix_t mat, double (*func)(double));

#endif // ML_MATH_H
