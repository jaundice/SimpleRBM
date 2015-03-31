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
            //return 1d / (1d + math.Exp(-x));


            if (x > 45)
                return 1;
            if (x < -45)
                return 0;

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
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    input[i, j] = HyperbolicTanValueD(input[i, j]);
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void SoftPlusInPlaceD(GThread thread, TElement[,] input)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    input[i, j] = math.Log(1 + math.Exp(input[i, j]));
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void LogisticD(GThread thread, TElement[,] input, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    output[i, j] = LogisticValueD(input[i, j]);
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void LogisticInPlaceD(GThread thread, TElement[,] input)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    input[i, j] = LogisticValueD(input[i, j]);
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void ExponentsD(GThread thread, TElement[,] input, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    output[i, j] = math.Exp(input[i, j]);
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
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