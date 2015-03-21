using System;
using Cudafy;
using TElement = System.Double;
using math = System.Math;
namespace SimpleRBM.Cuda
{
    public static partial class ActivationFunctionsCuda
    {
        [Cudafy]
        public static TElement LogisticValueD(TElement x)
        {
            return 1d / (1d + math.Exp(-x));
        }

        [Cudafy]
        public static TElement HyperbolicTanValueD(TElement x)
        {
            return math.Tanh(x);
        }


        [Cudafy]
        public static void TanhInPlaceD(GThread thread, TElement[,] input)
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
                    input[i, n] = HyperbolicTanValueD(input[i, n]);

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void SoftPlusInPlaceD(GThread thread, TElement[,] input)
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
                    input[i, n] = math.Log(1 + math.Exp(input[i, n]));

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void LogisticD(GThread thread, TElement[,] input, TElement[,] output)
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

        [Cudafy]
        public static void LogisticInPlaceD(GThread thread, TElement[,] input)
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
                    input[i, n] = LogisticValueD(input[i, n]);

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void ExponentsD(GThread thread, TElement[,] input, TElement[,] output)
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
                    output[i, n] = math.Exp(input[i, n]);

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void LogSumOfExponentsD(GThread thread, TElement[,] input, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            int x = input.GetLength(0);
            int y = input.GetLength(1);

            while (i < x)
            {
                TElement max = 0.0d;

                for (int jindex = 0; jindex < y; jindex++)
                {
                    max = math.Max(max, input[i, jindex]);
                }

                for (int k = 0; k < y; k++)
                {
                    output[i, k] = math.Exp(input[i, k] - max);
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }
    }
}