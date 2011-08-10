/*
 * Copyright (C) 2011 by Justin Holewinski
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

extern "C"
void matrix_multiply(float* A,
                     float* B,
                     float* C)
{
  unsigned globalX = 
    __builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x()
    + __builtin_ptx_read_tid_x();
  unsigned globalY = 
    __builtin_ptx_read_ctaid_y() * __builtin_ptx_read_ntid_y()
    + __builtin_ptx_read_tid_y();
  unsigned size    = 
    __builtin_ptx_read_nctaid_x() * __builtin_ptx_read_ntid_x();
  
  unsigned k;
  float    sum = 0.0f;
  
  for(k = 0; k < size; ++k)
  {
    sum += A[globalY * size + k] * B[k * size + globalX];
  }

  C[globalY * size + globalX] = sum;
}
