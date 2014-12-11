using System;
using System.Threading.Tasks;
using Cudafy;

namespace SimpleRBM.Cuda
{
    public static class Matrix2DCudaF
    {
        public const uint TRUE = 1u;
        public const uint FALSE = 0u;

        [Cudafy]
        public static void TransposeF(GThread thread, float[,] input, float[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input.GetLength(0))
            {
                int n = j;
                while (n < input.GetLength(1))
                {
                    output[n, i] = input[i, n];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void MultiplyScalarF(GThread thread, float[,] input, float factor, float[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input.GetLength(0))
            {
                int n = j;

                while (n < input.GetLength(1))
                {
                    output[i, n] = input[i, n] * factor;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void DivideF(GThread thread, float[,] input, float denominator, float[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            float factor = 1.0f / denominator;

            while (i < input.GetLength(0))
            {
                int n = j;
                while (n < input.GetLength(1))
                {
                    output[i, n] = input[i, n] * factor;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void MultiplyF(GThread thread, float[,] input1, float[,] input2, float[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input1.GetLength(0))
            {
                int n = j;
                while (n < input2.GetLength(1))
                {
                    output[i, n] = MultiplyElementF(input1, input2, i, n);
                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void IncrementF(GThread thread, float[,] input)
        {
            int rowIdx = thread.threadIdx.x + (thread.blockIdx.x * thread.blockDim.x);
            int colIdx = thread.threadIdx.y + (thread.blockIdx.y * thread.blockDim.y);


            int row = rowIdx;
            while (row < input.GetLength(0))
            {
                int col = colIdx;
                while (col < input.GetLength(1))
                {
                    input[row, col] = input[row, col] + 1.0f;
                    col += thread.gridDim.y * thread.blockDim.y;
                }
                row += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        private static float MultiplyElementF(float[,] A, float[,] B, int y, int x)
        {
            int aCols = A.GetLength(1);
            float accumulate = 0;
            for (int xx = 0; xx < aCols; xx++)
            {
                accumulate += A[y, xx] * B[xx, x];
            }

            return accumulate;
        }

        [Cudafy]
        public static void FillF(GThread thread, float[,] matrix, float value)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix.GetLength(0))
            {
                int n = j;
                while (n < matrix.GetLength(1))
                {
                    matrix[i, n] = value;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void OnesF(GThread thread, float[,] matrix)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix.GetLength(0))
            {
                int n = j;
                while (n < matrix.GetLength(1))
                {
                    matrix[i, n] = 1.0f;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void ZerosF(GThread thread, float[,] matrix)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix.GetLength(0))
            {
                int n = j;
                while (n < matrix.GetLength(1))
                {
                    matrix[i, n] = 0.0f;
                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void InsertValuesFromF(GThread thread, float[,] target, int mPos, int nPos, float[,] src,
            int mSize, int nSize)
        {
            mSize = mSize == 0 ? target.GetLength(0) - mPos : mSize;
            nSize = nSize == 0 ? target.GetLength(1) - nPos : nSize;


            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < mSize)
            {
                int n = j;
                while (n < nSize)
                {
                    target[i + mPos, n + nPos] = src[i, n];
                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void GreaterThanF(GThread thread, float[,] matrix1, float[,] matrix2, float[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                int n = j;
                while (n < matrix1.GetLength(1))
                {
                    output[i, n] = matrix1[i, n] > matrix2[i, n] ? 1.0f : 0.0f;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void LessThanF(GThread thread, float[,] matrix1, float[,] matrix2, float[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                int n = j;
                while (n < matrix1.GetLength(1))
                {
                    output[i, n] = matrix1[i, n] < matrix2[i, n] ? 1.0f : 0.0f;
                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void SubMatrixF(GThread thread, float[,] matrix, int startRow, int startCol, int numRows,
            int numCols, float[,] target)
        {
            numRows = numRows != 0 ? numRows : matrix.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : matrix.GetLength(1) - startCol;

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < numRows)
            {
                int n = j;
                while (n < numCols)
                {
                    target[i, n] = matrix[i + startRow, n + startCol];
                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void ToVectorF(GThread thread, float[,] matrix, float[,] target)
        {
            if (matrix.GetLength(1) == 1)
            {
                int numRows = matrix.GetLength(0);
                int numCols = 1;

                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

                while (i < numRows)
                {
                    int n = j;
                    while (n < numCols)
                    {
                        target[i, n] = matrix[i, n];

                        n += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                }
            }
            if (matrix.GetLength(0) == 1)
            {
                int numRows = 1;
                int numCols = matrix.GetLength(1);

                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

                while (i < numRows)
                {
                    int n = j;
                    while (n < numCols)
                    {
                        target[i, n] = matrix[i, n];

                        n += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                }
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void InsertValuesFromRowOrColumnF(GThread thread, float[,] target, float[,] source, int length,
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

                        i += thread.gridDim.x * thread.blockDim.x;
                    }
                }
                else
                {
                    while (i < length)
                    {
                        target[mPos, nPos + i] = source[0, i];

                        i += thread.gridDim.x * thread.blockDim.x;
                    }
                }
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void UpdateValueAlongAxisF(GThread thread, float[,] matrix, int index, float value, uint isRow)
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
        public static void AddF(GThread thread, float[,] matrix1, float[,] matrix2, float[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                int n = j;
                while (n < matrix1.GetLength(1))
                {
                    target[i, n] = matrix1[i, n] + matrix2[i, n];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void SubtractF(GThread thread, float[,] matrix1, float[,] matrix2, float[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                int n = j;
                while (n < matrix1.GetLength(1))
                {
                    target[i, n] = matrix1[i, n] - matrix2[i, n];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void PowF(GThread thread, float[,] matrix1, float power, float[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            if (power == 2)
            {
                while (i < matrix1.GetLength(0))
                {
                    int n = j;
                    while (n < matrix1.GetLength(1))
                    {
                        target[i, n] = matrix1[i, n] * matrix1[i, n];

                        n += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
            else
            {
                while (i < matrix1.GetLength(0))
                {
                    int n = j;
                    while (n < matrix1.GetLength(1))
                    {
                        target[i, n] = GMath.Pow(matrix1[i, n], power);

                        n += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
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

        [Cudafy]
        public static void IdentityF(GThread thread, float[,] matrix1)
        {
            int rowidx = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int colidx = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            for (var row = rowidx; row < matrix1.GetLength(0); row += thread.gridDim.x * thread.blockDim.x)
            {
                for (int col = colidx; col < matrix1.GetLength(1); col += thread.gridDim.y * thread.blockDim.y)
                {
                    var d = (row == col) ? 1.0f : 0.0f;
                    matrix1[row, col] = d;
                }
            }
        }

        [Cudafy]
        public static void ToBinaryF(GThread thread, float[,] matrix)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix.GetLength(0))
            {
                int n = j;
                while (n < matrix.GetLength(1))
                {
                    matrix[i, n] = matrix[i, n] < 0.5f ? 0f : 1f;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
        }
    }


   
}