using System;
using System.Threading.Tasks;
using Cudafy;

namespace CudaRbm
{
    public static class Matrix2DD
    {
        public const uint TRUE = 1u;
        public const uint FALSE = 0u;

        [Cudafy]
        public static void Transpose(GThread thread, double[,] input, double[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input.GetLength(0))
            {
                while (j < input.GetLength(1))
                {
                    output[j, i] = input[i, j];

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void MultiplyScalar(GThread thread, double[,] input, double factor, double[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input.GetLength(0))
            {
                while (j < input.GetLength(1))
                {
                    output[i, j] = input[i, j] * factor;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void Divide(GThread thread, double[,] input, double denominator, double[,] output)
        {
            //MultiplyScalar(thread, input, 1/denominator, output);

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            var factor = 1.0 / denominator;

            while (i < input.GetLength(0))
            {
                while (j < input.GetLength(1))
                {
                    output[i, j] = input[i, j] * factor;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void Multiply(GThread thread, double[,] input1, double[,] input2, double[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input1.GetLength(0))
            {
                while (j < input2.GetLength(1))
                {
                    output[i, j] = MultiplyElement(input1, input2, i, j);

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        private static double MultiplyElement(double[,] A, double[,] B, int y, int x)
        {
            int aCols = A.GetLength(1);
            double accumulate = 0;
            for (int xx = 0; xx < aCols; xx++)
            {
                accumulate += A[y, xx] * B[xx, x];
            }

            return accumulate;
        }

        [Cudafy]
        public static void Fill(GThread thread, double[,] matrix, double value)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix.GetLength(0))
            {
                while (j < matrix.GetLength(1))
                {
                    matrix[i, j] = value;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void Ones(GThread thread, double[,] matrix)
        {
            //Fill(thread, matrix, 1.0);

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix.GetLength(0))
            {
                while (j < matrix.GetLength(1))
                {
                    matrix[i, j] = 1.0;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void Zeros(GThread thread, double[,] matrix)
        {
            //Fill(thread, matrix, 0.0);
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix.GetLength(0))
            {
                while (j < matrix.GetLength(1))
                {
                    matrix[i, j] = 0.0;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void InsertValuesFrom(GThread thread, double[,] target, int mPos, int nPos, double[,] src,
            int mSize, int nSize)
        {
            mSize = mSize == 0 ? target.GetLength(0) - mPos : mSize;
            nSize = nSize == 0 ? target.GetLength(1) - nPos : nSize;


            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < mSize)
            {
                while (j < nSize)
                {
                    target[i + mPos, j + nPos] = src[i, j];

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void GreaterThan(GThread thread, double[,] matrix1, double[,] matrix2, double[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                while (j < matrix1.GetLength(1))
                {
                    output[i, j] = matrix1[i, j] > matrix2[i, j] ? 1.0 : 0.0;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void LessThan(GThread thread, double[,] matrix1, double[,] matrix2, double[,] output)
        {
            //GreaterThan(thread, matrix2, matrix1, output);

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                while (j < matrix1.GetLength(1))
                {
                    output[i, j] = matrix1[i, j] < matrix2[i, j] ? 1.0 : 0.0;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void SubMatrix(GThread thread, double[,] matrix, int startRow, int startCol, int numRows,
            int numCols, double[,] target)
        {
            numRows = numRows != 0 ? numRows : matrix.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : matrix.GetLength(1) - startCol;

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < numRows)
            {
                while (j < numCols)
                {
                    target[i, j] = matrix[i + startRow, j + startCol];

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void ToVector(GThread thread, double[,] matrix, double[,] target)
        {
            if (matrix.GetLength(1) == 1)
            {
                //SubMatrix(thread, matrix, 0, 0, 0, 1, target);

                var numRows = matrix.GetLength(0);
                var numCols = 1;

                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

                while (i < numRows)
                {
                    while (j < numCols)
                    {
                        target[i, j] = matrix[i, j];

                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                    thread.SyncThreadsCount(true);
                }
            }
            if (matrix.GetLength(0) == 1)
            {
                //SubMatrix(thread, matrix, 0, 0, 1, 0, target);

                var numRows = 1;
                var numCols = matrix.GetLength(1);

                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

                while (i < numRows)
                {
                    while (j < numCols)
                    {
                        target[i, j] = matrix[i, j];

                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                    thread.SyncThreadsCount(true);
                }
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void InsertValuesFromRowOrColumn(GThread thread, double[,] target, double[,] source, int length,
            uint fromColumn, int mPos, int nPos)
        {
            length = length == 0 ? Math.Max(source.GetLength(0), source.GetLength(1)) : length;


            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            if (j == nPos)
            {
                if (fromColumn == TRUE)
                {
                    while (i < length)
                    {
                        target[i + mPos, nPos] = source[i, 0];

                        j += thread.gridDim.y * thread.blockDim.y;

                        i += thread.gridDim.x * thread.blockDim.x;
                    }
                }
                else
                {
                    while (i < length)
                    {
                        target[mPos, nPos + i] = source[0, i];

                        j += thread.gridDim.y * thread.blockDim.y;

                        i += thread.gridDim.x * thread.blockDim.x;
                    }
                }
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void UpdateValueAlongAxis(GThread thread, double[,] matrix, int index, double value, uint isRow)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            if (isRow == TRUE)
            {
                if (j == index)
                {
                    while (i < matrix.GetLength(1))
                    {
                        matrix[index, i] = value;

                        //j += thread.gridDim.y * thread.blockDim.y;

                        i += thread.gridDim.x * thread.blockDim.x;
                    }
                }
            }
            else
            {
                if (i == index)
                {
                    while (j < matrix.GetLength(0))
                    {
                        matrix[j, index] = value;

                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                }
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void Add(GThread thread, double[,] matrix1, double[,] matrix2, double[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                while (j < matrix1.GetLength(1))
                {
                    target[i, j] = matrix1[i, j] + matrix2[i, j];

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void Subtract(GThread thread, double[,] matrix1, double[,] matrix2, double[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                while (j < matrix1.GetLength(1))
                {
                    target[i, j] = matrix1[i, j] - matrix2[i, j];

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
                thread.SyncThreadsCount(true);
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void Pow(GThread thread, double[,] matrix1, double power, double[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            if (power == 2)
            {
                while (i < matrix1.GetLength(0))
                {
                    while (j < matrix1.GetLength(1))
                    {
                        target[i, j] = matrix1[i, j] * matrix1[i, j];

                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                    thread.SyncThreadsCount(true);
                }
            }
            else
            {
                while (i < matrix1.GetLength(0))
                {
                    while (j < matrix1.GetLength(1))
                    {
                        target[i, j] = Math.Pow(matrix1[i, j], power);

                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                    thread.SyncThreadsCount(true);
                }
            }

            thread.SyncThreads();
        }

        public static T[,] JaggedToMultidimesional<T>(T[][] source)
        {
            int rows = source.GetLength(0);
            int cols = source[0].GetLength(0);

            var res = new T[rows, cols];
            Parallel.For(0, rows, i => Parallel.For(
                0, cols, j => { res[i, j] = source[i][j]; }));

            return res;
        }
    }
}