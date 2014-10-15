using System;
using System.Runtime.CompilerServices;
using Cudafy;

namespace CudaRbm
{
    public static class ActivationFunctions
    {
        [Cudafy]
        public static float LogisticValue(float x)
        {
            return 1f / (1f + GMath.Exp(-x));
        }

        [Cudafy]
        public static void Logistic(GThread thread, float[,] input, float[,] output)
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
                    output[i, n] = LogisticValue(input[i, n]);

                    //output[i, n] = 1f;

                    //output[i, n] = 1f / (1f + GMath.Exp(-1 * input[i, n]));

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }
    }
}