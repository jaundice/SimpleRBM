using Cudafy;
using TElement =System.Single;
using math = Cudafy.GMath;

namespace SimpleRBM.Cuda
{
    public static partial class ActivationFunctionsCuda
    {
        [Cudafy]
        public static TElement LogisticValueF(TElement x)
        {
            return 1f / (1f + math.Exp(-x));
        }

        [Cudafy]
        public static TElement HyperbolicTanValueF(TElement x)
        {
            return math.Tanh(x);
        }


        [Cudafy]
        public static void TanhInPlaceF(GThread thread, TElement[,] input)
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
                    input[i, n] = HyperbolicTanValueF(input[i, n]);

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void SoftPlusInPlaceF(GThread thread, TElement[,] input)
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
        public static void LogisticF(GThread thread, TElement[,] input, TElement[,] output)
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
        public static void LogisticInPlaceF(GThread thread, TElement[,] input)
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
                    input[i, n] = LogisticValueF(input[i, n]);

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void ExponentsF(GThread thread, TElement[,] input, TElement[,] output)
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
        public static void LogSumOfExponentsF(GThread thread, TElement[,] input, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            int x = input.GetLength(0);
            int y = input.GetLength(1);

            while (i < x)
            {
                TElement max = 0.0f;

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