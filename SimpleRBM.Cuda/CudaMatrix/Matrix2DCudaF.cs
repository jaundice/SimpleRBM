using System;
using System.Threading;
using System.Threading.Tasks;
using Cudafy;
using TElementType = System.Single;
using math = Cudafy.GMath;

namespace SimpleRBM.Cuda
{
    public static partial class Matrix2DCuda
    {
        [Cudafy]
        public static void MaximumElementValueRowWiseF(GThread thread, TElementType[,] input, TElementType[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            //int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input.GetLength(0))
            {
                TElementType max = 0;
                for (var j = 0; j < input.GetLength(1); j++)
                {
                    max = math.Max(max, input[i, j]);
                }
                output[i,0] = max;
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        [Cudafy]
        public static void TransposeF(GThread thread, TElementType[,] input, TElementType[,] output)
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
        public static void MultiplyScalarF(GThread thread, TElementType[,] input, TElementType factor, TElementType[,] output)
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
        public static void MultiplyScalarInPlaceF(GThread thread, TElementType[,] input, TElementType factor)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input.GetLength(0))
            {
                int n = j;

                while (n < input.GetLength(1))
                {
                    input[i, n] = input[i, n] * factor;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void DivideF(GThread thread, TElementType[,] input, TElementType denominator, TElementType[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            TElementType factor = 1.0f / denominator;

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

        /// <summary>
        ///     divides each element of input by the element in the first column of the matching row in denominator
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="input"></param>
        /// <param name="denominator"></param>
        /// <param name="output"></param>
        [Cudafy]
        public static void DivideByF(GThread thread, TElementType[,] input, TElementType[,] denominator, TElementType[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < input.GetLength(0))
            {
                int n = j;
                while (n < input.GetLength(1))
                {
                    output[i, n] = input[i, n] / denominator[i, 0];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void MultiplyF(GThread thread, TElementType[,] input1, TElementType[,] input2, TElementType[,] output)
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
        public static void IncrementF(GThread thread, TElementType[,] input)
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
        private static TElementType MultiplyElementF(TElementType[,] A, TElementType[,] B, int y, int x)
        {
            int aCols = A.GetLength(1);
            TElementType accumulate = 0;
            for (int xx = 0; xx < aCols; xx++)
            {
                accumulate += A[y, xx] * B[xx, x];
            }

            return accumulate;
        }

        [Cudafy]
        public static void FillF(GThread thread, TElementType[,] matrix, TElementType value)
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
        public static void OnesF(GThread thread, TElementType[,] matrix)
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
        public static void ZerosF(GThread thread, TElementType[,] matrix)
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
        public static void InsertValuesFromF(GThread thread, TElementType[,] target, int mPos, int nPos, TElementType[,] src,
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
        public static void GreaterThanF(GThread thread, TElementType[,] matrix1, TElementType[,] matrix2, TElementType[,] output)
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
        public static void GreaterThanLinearF(GThread thread, TElementType[,] matrix1, TElementType[,] matrix2, TElementType[,] output)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                int n = j;
                while (n < matrix1.GetLength(1))
                {
                    output[i, n] = matrix1[i, n] > matrix2[i, n] ? matrix1[i, n] : 0.0f;

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }

            thread.SyncThreads();
        }

        [Cudafy]
        public static void LessThanF(GThread thread, TElementType[,] matrix1, TElementType[,] matrix2, TElementType[,] output)
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
        public static void SubMatrixF(GThread thread, TElementType[,] matrix, int startRow, int startCol, int numRows,
            int numCols, TElementType[,] target)
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
        public static void ToVectorF(GThread thread, TElementType[,] matrix, TElementType[,] target)
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
        public static void InsertValuesFromRowOrColumnF(GThread thread, TElementType[,] target, TElementType[,] source, int length,
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
        public static void UpdateValueAlongAxisF(GThread thread, TElementType[,] matrix, int index, TElementType value, uint isRow)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            if (isRow == TRUE)
            {
                //if (j == index)
                //{
                while (i < matrix.GetLength(1))
                {
                    matrix[index, i] = value;

                    i += thread.gridDim.x * thread.blockDim.x;
                }
                //}
            }
            else
            {
                //if (i == index)
                //{
                while (j < matrix.GetLength(0))
                {
                    matrix[j, index] = value;

                    j += thread.gridDim.y * thread.blockDim.y;
                }
                //}
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void AddF(GThread thread, TElementType[,] matrix1, TElementType[,] matrix2, TElementType[,] target)
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
        public static void AddInPlaceF(GThread thread, TElementType[,] matrix1, TElementType[,] matrix2)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                int n = j;
                while (n < matrix1.GetLength(1))
                {
                    matrix1[i, n] = matrix1[i, n] + matrix2[i, n];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void UpdateWithMomentumF(GThread thread, TElementType[,] oldValues, TElementType[,] newValues, TElementType[,] target,
            TElementType momentum)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < oldValues.GetLength(0))
            {
                int n = j;
                while (n < oldValues.GetLength(1))
                {
                    target[i, n] = oldValues[i, n] + (momentum * (newValues[i, n] - oldValues[i, n]));

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void UpdateWithMomentumInPlaceF(GThread thread, TElementType[,] oldValues, TElementType[,] newValues,
            TElementType momentum)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < oldValues.GetLength(0))
            {
                int n = j;
                while (n < oldValues.GetLength(1))
                {
                    oldValues[i, n] = oldValues[i, n] + (momentum * (newValues[i, n] - oldValues[i, n]));

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void SubtractF(GThread thread, TElementType[,] matrix1, TElementType[,] matrix2, TElementType[,] target)
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
        public static void SubtractInPlaceF(GThread thread, TElementType[,] matrix1, TElementType[,] matrix2)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < matrix1.GetLength(0))
            {
                int n = j;
                while (n < matrix1.GetLength(1))
                {
                    matrix1[i, n] = matrix1[i, n] - matrix2[i, n];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void PowF(GThread thread, TElementType[,] matrix1, TElementType power, TElementType[,] target)
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
                        target[i, n] = math.Pow(matrix1[i, n], power);

                        n += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
        }

        [Cudafy]
        public static void PowInPlaceF(GThread thread, TElementType[,] matrix1, TElementType power)
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
                        matrix1[i, n] = matrix1[i, n] * matrix1[i, n];

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
                        matrix1[i, n] = math.Pow(matrix1[i, n], power);

                        n += thread.gridDim.y * thread.blockDim.y;
                    }
                    i += thread.gridDim.x * thread.blockDim.x;
                }
                thread.SyncThreads();
            }
        }

        [Cudafy]
        public static void IdentityF(GThread thread, TElementType[,] matrix1)
        {
            int rowidx = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int colidx = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;
            for (int row = rowidx; row < matrix1.GetLength(0); row += thread.gridDim.x * thread.blockDim.x)
            {
                for (int col = colidx; col < matrix1.GetLength(1); col += thread.gridDim.y * thread.blockDim.y)
                {
                    TElementType d = (row == col) ? 1.0f : 0.0f;
                    matrix1[row, col] = d;
                }
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void ToBinaryF(GThread thread, TElementType[,] matrix)
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
            thread.SyncThreads();
        }

        /// <summary>
        /// Fills a column vector, reduced, with the sum of each element in the corresponding row of matrix
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="matrix"></param>
        /// <param name="reduced"></param>
        [Cudafy]
        public static void SumMatrixRowsF(GThread thread, TElementType[,] matrix, TElementType[,] reduced)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (i < matrix.GetLength(0))
            {
                TElementType sum = 0f;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j];
                }
                reduced[i, 0] = sum;
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }


        /// <summary>
        /// Fills a row vector, reduced, with the sum of each element in the corresponding column or matrix
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="matrix"></param>
        /// <param name="reduced"></param>
        [Cudafy]
        public static void SumMatrixColumnsF(GThread thread, TElementType[,] matrix, TElementType[,] reduced)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (i < matrix.GetLength(1))
            {
                TElementType sum = 0f;
                for (int j = 0; j < matrix.GetLength(0); j++)
                {
                    sum += matrix[j, i];
                }
                reduced[0, i] = sum;
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void CopyToArrayAtNF(GThread thread, TElementType[,] target, TElementType[] source, int x)
        {
            int id = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (id < source.GetLength(0))
            {
                target[x, id] = source[id];
                id += thread.blockDim.x * thread.gridDim.x;
            }

            thread.SyncThreads();
        }


        [Cudafy]
        public static void CopyToArrayAtNF2(GThread thread, TElementType[,] target, TElementType[] source, TElementType scale)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < target.GetLength(0))
            {
                int n = j;
                while (n < target.GetLength(1))
                {
                    target[i, n] = scale * source[i * target.GetLength(1) + n];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }
        [Cudafy]
        public static void RepMatRowsF(GThread thread, TElementType[,] source, TElementType[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < target.GetLength(0))
            {
                int n = j;
                while (n < target.GetLength(1))
                {
                    target[i, n] = source[0, n];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }

        [Cudafy]
        public static void RepMatColsF(GThread thread, TElementType[,] source, TElementType[,] target)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int j = thread.threadIdx.y + thread.blockIdx.y * thread.blockDim.y;

            while (i < target.GetLength(0))
            {
                int n = j;
                while (n < target.GetLength(1))
                {
                    target[i, n] = source[i, 0];

                    n += thread.gridDim.y * thread.blockDim.y;
                }
                i += thread.gridDim.x * thread.blockDim.x;
            }
            thread.SyncThreads();
        }
    }
}