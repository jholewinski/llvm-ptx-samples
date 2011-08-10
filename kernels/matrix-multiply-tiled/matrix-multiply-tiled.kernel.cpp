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

// This must be changed to reflect changes in matrix-multiply-tiled.cpp
#define BLOCK_SIZE 16

__attribute__((address_space(4)))
float g_scratchA[BLOCK_SIZE][BLOCK_SIZE];
__attribute__((address_space(4)))
float g_scratchB[BLOCK_SIZE][BLOCK_SIZE];

extern "C"
void matrix_multiply_tiled(float* A,
                           float* B,
                           float* C)
{
  int   globalX   = __builtin_ptx_read_ctaid_x() * __builtin_ptx_read_ntid_x() + __builtin_ptx_read_tid_x();
  int   globalY   = __builtin_ptx_read_ctaid_y() * __builtin_ptx_read_ntid_y() + __builtin_ptx_read_tid_y();
  int   size      = __builtin_ptx_read_nctaid_x() * __builtin_ptx_read_ntid_x();
  int   k;
  float sum       = 0.0f;
  int   numBlocks = size / BLOCK_SIZE;
  int   b;

  int tidX = __builtin_ptx_read_tid_x();
  int tidY = __builtin_ptx_read_tid_y();

  for(b = 0; b < numBlocks; ++b)
  {
    // Populate a cache for A/B
    int x;
    int y;

    x = b * BLOCK_SIZE + tidX;
    y = globalY;
    
    g_scratchA[tidY][tidX] = A[y * size + x];

    x = globalX;
    y = b * BLOCK_SIZE + tidY;

    g_scratchB[tidY][tidX] = B[y * size + x];
    
    __builtin_ptx_bar_sync(0);
    
    for(k = 0; k < BLOCK_SIZE; ++k)
    {
      float myA;
      float myB;

      myA = g_scratchA[tidY][k];
      myB = g_scratchB[k][tidX];

      sum += myA * myB;
    }

    __builtin_ptx_bar_sync(0);
  }
  
  C[globalY * size + globalX] = sum;
}
