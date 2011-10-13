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

// This must match BLOCK_SIZE in matmul.cpp
#define BLOCK_SIZE 16

__kernel
void matmul(__global float* A, __global float* B, __global float* C) {

  __local float scratchA[BLOCK_SIZE][BLOCK_SIZE];
  __local float scratchB[BLOCK_SIZE][BLOCK_SIZE];

  int   globalX   = get_global_id(0);
  int   globalY   = get_global_id(1);
  int   size      = get_global_size(0);
  int   k;
  float sum       = 0.0f;
  int   numBlocks = size / BLOCK_SIZE;
  int   b;

  int tidX = get_local_id(0);
  int tidY = get_local_id(1);

  for(b = 0; b < numBlocks; ++b)
  {
    // Populate a cache for A/B
    int x;
    int y;

    x = b * BLOCK_SIZE + tidX;
    y = globalY;

    scratchA[tidY][tidX] = A[y * size + x];

    x = globalX;
    y = b * BLOCK_SIZE + tidY;

    scratchB[tidY][tidX] = B[y * size + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(k = 0; k < BLOCK_SIZE; ++k)
    {
      float myA;
      float myB;

      myA = scratchA[tidY][k];
      myB = scratchB[k][tidX];

      sum += myA * myB;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[globalY * size + globalX] = sum;

}
