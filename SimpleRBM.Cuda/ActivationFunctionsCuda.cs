using System;
using Cudafy;

namespace SimpleRBM.Cuda
{
    public static class ActivationFunctionsCuda
    {
        [Cudafy]
        public static float LogisticValueF(float x)
        {
            return 1f / (1f + GMath.Exp(-x));
        }

        [Cudafy]
        public static void LogisticF(GThread thread, float[,] input, float[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            int x = input.GetLength(0);
            int y = input.GetLength(1);

            while (i < x)
            {
                int n = j;
                while (n < y)
                {
                    output[i, n] = LogisticValueF(input[i, n]);

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static double LogisticValueD(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        } 
       
        [Cudafy]
        public static void LogisticD(GThread thread, double[,] input, double[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            int x = input.GetLength(0);
            int y = input.GetLength(1);

            while (i < x)
            {
                int n = j;
                while (n < y)
                {
                    output[i, n] = LogisticValueD(input[i, n]);

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }
    }
}