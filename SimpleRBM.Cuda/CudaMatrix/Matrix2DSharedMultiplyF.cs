using System;
using System.Diagnostics;
using Cudafy;
using Mono.CSharp;

namespace SimpleRBM.Cuda
{
    public partial class Matrix2DCuda
    {
        //[Cudafy]
        //public unsafe static void GEMMD(GThread thread, double[,] mA, double[,] mB, double[,] mC, double alpha = 1.0,
        //    double beta = 1.0)
        //{

        //    int lda = mA.GetLength(1);
        //    int ldb = mB.GetLength(1);
        //    int ldc = mC.GetLength(1);
        //    int k = mA.GetLength(0);

        //    fixed (double* aptr = mA)
        //    fixed (double* bptr = mB)
        //    fixed (double* cptr = mC)
        //    {
        //        var A = aptr + thread.blockIdx.x * 64 + thread.threadIdx.x + thread.threadIdx.y * 16;
        //        var B = bptr + thread.threadIdx.x + (thread.blockIdx.y * 16 + thread.threadIdx.y) * ldb;
        //        var C = cptr + thread.blockIdx.x * 64 + thread.threadIdx.x + (thread.threadIdx.y + thread.blockIdx.y * ldc) * 16;
        //        double* bLast = B + k;

        //        double[,] bs = thread.AllocateShared<double>("shared", 16, 17);

        //        //double[, ,] c = thread.AllocateShared<double>("working", 16, 16, 16); // we can't allocate a local double[16] so allocate it shared
        //        double c0 = 0.0;
        //        double c1 = 0.0;
        //        double c2 = 0.0;
        //        double c3 = 0.0;
        //        double c4 = 0.0;
        //        double c5 = 0.0;
        //        double c6 = 0.0;
        //        double c7 = 0.0;
        //        double c8 = 0.0;
        //        double c9 = 0.0;
        //        double c10 = 0.0;
        //        double c11 = 0.0;
        //        double c12 = 0.0;
        //        double c13 = 0.0;
        //        double c14 = 0.0;
        //        double c15 = 0.0;

        //        do
        //        {
        //            //GThread.InsertCode("#pragma unroll");

        //            for (int i = 0; i < 16; i += 4)
        //            {
        //                bs[thread.threadIdx.x, thread.threadIdx.y + i] = 1.0;
        //                //bs[thread.threadIdx.x, thread.threadIdx.y + i] = B[i * ldb];
        //            }
        //            B += 16;
        //            thread.SyncThreads();

        //            //GThread.InsertCode("#pragma unroll");

        //            for (int i = 0; i < 16; i++, A += lda)
        //            {
        //                c0 += A[0] * bs[i, 0];
        //                c1 += A[0] * bs[i, 1];
        //                c2 += A[0] * bs[i, 2];
        //                c3 += A[0] * bs[i, 3];
        //                c4 += A[0] * bs[i, 4];
        //                c5 += A[0] * bs[i, 5];
        //                c6 += A[0] * bs[i, 6];
        //                c7 += A[0] * bs[i, 7];
        //                c8 += A[0] * bs[i, 8];
        //                c9 += A[0] * bs[i, 9];
        //                c10 += A[0] * bs[i, 10];
        //                c11 += A[0] * bs[i, 11];
        //                c12 += A[0] * bs[i, 12];
        //                c13 += A[0] * bs[i, 13];
        //                c14 += A[0] * bs[i, 14];
        //                c15 += A[0] * bs[i, 15];
        //            }

        //            thread.SyncThreads();

        //        } while (B < bLast);

        //        //for (int i = 0; i < 16; i++, C += ldc)
        //        //    C[0] = alpha * c[thread.threadIdx.x, thread.threadIdx.y, i] + beta * C[0];

        //        C[0 * ldc] = alpha * c0 + beta * C[0 * ldc];
        //        C[1 * ldc] = alpha * c1 + beta * C[1 * ldc];
        //        C[2 * ldc] = alpha * c2 + beta * C[2 * ldc];
        //        C[3 * ldc] = alpha * c3 + beta * C[3 * ldc];
        //        C[4 * ldc] = alpha * c4 + beta * C[4 * ldc];
        //        C[5 * ldc] = alpha * c5 + beta * C[5 * ldc];
        //        C[6 * ldc] = alpha * c6 + beta * C[6 * ldc];
        //        C[7 * ldc] = alpha * c7 + beta * C[7 * ldc];
        //        C[8 * ldc] = alpha * c8 + beta * C[8 * ldc];
        //        C[9 * ldc] = alpha * c9 + beta * C[9 * ldc];
        //        C[10 * ldc] = alpha * c10 + beta * C[10 * ldc];
        //        C[11 * ldc] = alpha * c11 + beta * C[11 * ldc];
        //        C[12 * ldc] = alpha * c12 + beta * C[12 * ldc];
        //        C[13 * ldc] = alpha * c13 + beta * C[13 * ldc];
        //        C[14 * ldc] = alpha * c14 + beta * C[14 * ldc];
        //        C[15 * ldc] = alpha * c15 + beta * C[15 * ldc];


        //        thread.SyncThreads();

        //    }





        //}
    }
}