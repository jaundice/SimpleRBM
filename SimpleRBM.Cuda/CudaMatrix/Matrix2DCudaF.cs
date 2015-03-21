using System;
using System.Threading;
using System.Threading.Tasks;
using Cudafy;
using TElement = System.Single;
using math = Cudafy.GMath;

namespace SimpleRBM.Cuda
{
    public static partial class Matrix2DCuda
    {

        [Cudafy]
        public static void MaximumElementValueRowWiseF(GThread thread, TElement[,] input, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (i < input.GetLength(0))
            {
                TElement max = 0;
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    max = math.Max(max, input[i, j]);
                }
                output[i, 0] = max;
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void TransposeF(GThread thread, TElement[,] input, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    output[j, i] = input[i, j];
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        /// <summary>
        ///     divides each element of input by the element in the first column of the matching row in denominator
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="input"></param>
        /// <param name="denominator"></param>
        /// <param name="output"></param>
        [Cudafy]
        public static void DivideByF(GThread thread, TElement[,] input, TElement[,] denominator, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    output[i, j] = input[i, j] / denominator[i, 0];
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void IncrementF(GThread thread, TElement[,] input)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < input.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < input.GetLength(1))
                {
                    thread.SyncThreads();
                    input[i, j] = input[i, j] + 1.0f;
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void FillF(GThread thread, TElement[,] matrix, TElement value)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < matrix.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < matrix.GetLength(1))
                {
                    thread.SyncThreads();
                    matrix[i, j] = value;
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void InsertValuesFromF(GThread thread, TElement[,] target, int mPos, int nPos, TElement[,] src,
            int mSize, int nSize)
        {
            mSize = mSize == 0 ? target.GetLength(0) - mPos : mSize;
            nSize = nSize == 0 ? target.GetLength(1) - nPos : nSize;

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < mSize)
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < nSize)
                {
                    thread.SyncThreads();
                    target[i + mPos, j + nPos] = src[i, j];
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void GreaterThanF(GThread thread, TElement[,] matrix1, TElement[,] matrix2, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < matrix1.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < matrix1.GetLength(1))
                {
                    thread.SyncThreads();
                    output[i, j] = matrix1[i, j] > matrix2[i, j] ? 1.0f : 0.0f;
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void GreaterThanLinearF(GThread thread, TElement[,] matrix1, TElement[,] matrix2,
            TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < matrix1.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < matrix1.GetLength(1))
                {
                    thread.SyncThreads();
                    output[i, j] = matrix1[i, j] > matrix2[i, j] ? matrix1[i, j] : 0.0f;
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void LessThanF(GThread thread, TElement[,] matrix1, TElement[,] matrix2, TElement[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < matrix1.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < matrix1.GetLength(1))
                {
                    thread.SyncThreads();
                    output[i, j] = matrix1[i, j] < matrix2[i, j] ? 1.0f : 0.0f;
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void SubMatrixF(GThread thread, TElement[,] matrix, int startRow, int startCol, int numRows,
            int numCols, TElement[,] target)
        {
            numRows = numRows != 0 ? numRows : matrix.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : matrix.GetLength(1) - startCol;

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < numRows)
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < numCols)
                {
                    thread.SyncThreads();
                    target[i, j] = matrix[i + startRow, j + startCol];
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void InsertValuesFromRowOrColumnF(GThread thread, TElement[,] target, TElement[,] source,
            int length,
            uint fromColumn, int mPos, int nPos)
        {
            length = length == 0 ? (int)math.Max(source.GetLength(0), source.GetLength(1)) : length;


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
        public static void UpdateValueAlongAxisF(GThread thread, TElement[,] matrix, int index, TElement value,
            uint isRow)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            if (isRow == TRUE)
            {
                while (i < matrix.GetLength(1))
                {
                    matrix[index, i] = value;

                    i += thread.gridDim.x * thread.blockDim.x;
                }
            }
            else
            {
                while (j < matrix.GetLength(0))
                {
                    matrix[j, index] = value;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void UpdateWithMomentumF(GThread thread, TElement[,] oldValues, TElement[,] newValues,
            TElement[,] target,
            TElement momentum)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < oldValues.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < oldValues.GetLength(1))
                {
                    thread.SyncThreads();
                    target[i, j] = oldValues[i, j] + (momentum * (newValues[i, j] - oldValues[i, j]));
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void UpdateWithMomentumInPlaceF(GThread thread, TElement[,] oldValues, TElement[,] newValues,
            TElement momentum)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < oldValues.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < oldValues.GetLength(1))
                {
                    thread.SyncThreads();
                    oldValues[i, j] = oldValues[i, j] + (momentum * (newValues[i, j] - oldValues[i, j]));
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void PowF(GThread thread, TElement[,] matrix1, TElement power, TElement[,] target)
        {
            if (power == 2)
            {
                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                while (i < matrix1.GetLength(0))
                {
                    int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                    while (j < matrix1.GetLength(1))
                    {
                        thread.SyncThreads();
                        target[i, j] = matrix1[i, j] * matrix1[i, j];
                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    thread.SyncThreads();
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
            else
            {
                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                while (i < matrix1.GetLength(0))
                {
                    int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                    while (j < matrix1.GetLength(1))
                    {
                        thread.SyncThreads();
                        target[i, j] = math.Pow(matrix1[i, j], power);
                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    thread.SyncThreads();
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
        }

        [Cudafy]
        public static void PowInPlaceF(GThread thread, TElement[,] matrix1, TElement power)
        {
            if (power == 2)
            {
                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                while (i < matrix1.GetLength(0))
                {
                    int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                    while (j < matrix1.GetLength(1))
                    {
                        thread.SyncThreads();
                        matrix1[i, j] = matrix1[i, j] * matrix1[i, j];
                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    thread.SyncThreads();
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
            else
            {
                int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
                while (i < matrix1.GetLength(0))
                {
                    int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                    while (j < matrix1.GetLength(1))
                    {
                        thread.SyncThreads();
                        matrix1[i, j] = math.Pow(matrix1[i, j], power);
                        j += thread.gridDim.y * thread.blockDim.y;
                    }
                    thread.SyncThreads();
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
        }


        [Cudafy]
        public static void IdentityF(GThread thread, TElement[,] matrix1)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < matrix1.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < matrix1.GetLength(1))
                {
                    thread.SyncThreads();
                    matrix1[i, j] = i == j ? 1.0f : 0.0f;
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void ToBinaryF(GThread thread, TElement[,] matrix)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            while (i < matrix.GetLength(0))
            {
                int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
                while (j < matrix.GetLength(1))
                {
                    thread.SyncThreads();
                    matrix[i, j] = (matrix[i, j] < 0.5f ? 0f : 1f);
                    j += thread.gridDim.y * thread.blockDim.y;
                }
                thread.SyncThreads();
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

    }
}