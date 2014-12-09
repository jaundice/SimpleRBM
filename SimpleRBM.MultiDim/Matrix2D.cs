using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace SimpleRBM.MultiDim
{
    public static class Matrix2D
    {
        public enum Axis
        {
            Horizontal = 0,
            Vertical = 1
        }

        public static float[,] Transpose(float[,] source)
        {
            int sourceRows = source.GetLength(0);
            int sourceCols = source.GetLength(1);

            var output = new float[sourceCols, sourceRows];

            for (int i = 0; i < sourceRows; i++)
            {
                for (int j = 0; j < sourceCols; j++)
                {
                    output[j, i] = source[i, j];
                }
            }
            return output;
        }

        public unsafe static double[,] Transpose(double[,] source)
        {
            int sourceRows = source.GetLength(0);
            int sourceCols = source.GetLength(1);

            var output = new double[sourceCols, sourceRows];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(output, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();


                Parallel.For(0, sourceRows, i =>
                    Parallel.For(0, sourceCols, j =>
                    {
                        UnsafeUpdate2DArray(arr, sourceRows, j, i, source[i, j]);
                        //output[j, i] = source[i, j];
                    }));

            }
            finally
            {
                handle.Free();
            }
            return output;
        }


        public static float[,] Multiply(float[,] source, float scalar)
        {
            int rows = source.GetLength(0);
            int cols = source.GetLength(1);

            var output = new float[rows, cols];

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    output[y, x] = source[y, x] * scalar;
                }
            }
            return output;
        }


        public static double[,] Divide(double[,] source, double denom)
        {
            return Multiply(source, 1 / denom);
        }

        public static unsafe double[,] Multiply(double[,] source, double scalar)
        {
            int rows = source.GetLength(0);
            int cols = source.GetLength(1);

            var output = new double[rows, cols];
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(output, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();

                Parallel.For(0, rows, y =>
                {
                    Parallel.For(0, cols, x =>
                    {
                        UnsafeUpdate2DArray(arr, cols, y, x, source[y, x] * scalar);
                        //output[y, x] = source[y, x]*scalar;
                    });
                });
            }
            finally
            {
                handle.Free();
            }
            return output;
        }


        public static float[,] Multiply(float[,] A, float[,] B)
        {
            int aRows = A.GetLength(0);
            int aCols = A.GetLength(1);

            int bRows = B.GetLength(0);
            int bCols = B.GetLength(1);

            if (aCols != bRows)
                throw new MatrixException("Invalid dimensions for Matrix Multiply");

            var output = new float[aRows, bCols];

            for (int y = 0; y < aRows; y++)
            {
                for (int x = 0; x < bCols; x++)
                {
                    output[y, x] = MultiplyElement(A, B, y, x);
                }
            }

            return output;
        }

        public static unsafe double[,] Multiply(double[,] A, double[,] B)
        {
            int aRows = A.GetLength(0);
            int aCols = A.GetLength(1);

            int bRows = B.GetLength(0);
            int bCols = B.GetLength(1);

            if (aCols != bRows)
                throw new MatrixException("Invalid dimensions for Matrix Multiply");

            var output = new double[aRows, bCols];
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(output, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();


                Parallel.For(0, aRows, row => Parallel.For(0, bCols, col =>
                {
                    UnsafeUpdate2DArray(arr, bCols, row, col, MultiplyElement(A, B, row, col));
                    //output[y, x] = MultiplyElement(A, B, y, x);
                }));
            }
            finally
            {
                handle.Free();
            }

            return output;
        }

        private static float MultiplyElement(float[,] A, float[,] B, int y, int x)
        {
            int aCols = A.GetLength(1);

            float accumulator = 0;
            for (int xx = 0; xx < aCols; xx++)
            {
                accumulator += A[y, xx] * B[xx, x];
            }
            return accumulator;
        }

        private static double MultiplyElement(double[,] A, double[,] B, int y, int x)
        {
            int aCols = A.GetLength(1);
            double accumulate =0;
            for (int xx = 0; xx < aCols; xx++)
            {
                accumulate += A[y, xx]*B[xx, x];
            }
           
            return accumulate;
        }


        public static float[,] IdentityF(int size)
        {
            var output = new float[size, size];
            for (int i = 0; i < size; i++)
            {
                output[i, i] = 1.0f;
            }
            return output;
        }

        public unsafe static double[,] IdentityD(int size)
        {
            var output = new double[size, size];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(output, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();

                Parallel.For(0, size, i =>
                {
                    UnsafeUpdate2DArray(arr, size, i, i, 1.0);
                });
            }
            finally
            {
                handle.Free();
            }
            return output;
        }


        public static float DeterminantF(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (rows != cols)
                throw new MatrixException("Matrix must be square");

            if (rows == 2)
            {
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            }

            float accumulate = 0f;
            int mult = 1;
            for (int i = 0; i < cols; i++)
            {
                var sub = new float[rows - 1, cols - 1];
                for (int y = 1; y < rows; y++)
                {
                    for (int x = 0; x < cols; x++)
                    {
                        if (x == i)
                            continue;

                        sub[y - 1, x < i ? x : x - 1] = matrix[y, x];
                    }
                }
                accumulate += matrix[0, i] * DeterminantF(sub) * mult;
                mult *= -1;
            }
            return accumulate;
        }

        public static double DeterminantD(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (rows != cols)
                throw new MatrixException("Matrix must be square");

            if (rows == 2)
            {
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            }

            double accumulate = 0f;
            int mult = 1;
            for (int i = 0; i < cols; i++)
            {
                var sub = new double[rows - 1, cols - 1];
                for (int y = 1; y < rows; y++)
                {
                    for (int x = 0; x < cols; x++)
                    {
                        if (x == i)
                            continue;

                        sub[y - 1, x < i ? x : x - 1] = matrix[y, x];
                    }
                }
                accumulate += matrix[0, i] * DeterminantD(sub) * mult;
                mult *= -1;
            }
            return accumulate;
        }



        public static double[,] Decompose(double[,] matrix, out int[] perm, out int toggle)
        {
            // Doolittle LUP decomposition.
            // assumes matrix is square.
            int n = matrix.GetLength(0); // convenience
            double[,] result = Duplicate(matrix, sizeof(double));
            perm = new int[n];
            for (int i = 0; i < n; ++i)
            {
                perm[i] = i;
            }
            toggle = 1;
            for (int j = 0; j < n - 1; ++j) // each column
            {
                double colMax = Math.Abs(result[j, j]); // largest val in col j
                int pRow = j;
                for (int i = j + 1; i < n; ++i)
                {
                    if (result[i, j] > colMax)
                    {
                        colMax = result[i, j];
                        pRow = i;
                    }
                }
                if (pRow != j) // swap rows
                {
                    SwapRows(matrix, pRow, j);
                    int tmp = perm[pRow]; // and swap perm info
                    perm[pRow] = perm[j];
                    perm[j] = tmp;
                    toggle = -toggle; // row-swap toggle
                }
                if (Math.Abs(result[j, j]) < 1.0E-20)
                    return null; // consider a throw
                for (int i = j + 1; i < n; ++i)
                {
                    result[i, j] /= result[j, j];
                    for (int k = j + 1; k < n; ++k)
                        result[i, k] -= result[i, j] * result[j, k];
                }
            } // main j column loop
            return result;
        }

        public static float[,] Decompose(float[,] matrix, out int[] perm, out int toggle)
        {
            // Doolittle LUP decomposition.
            // assumes matrix is square.
            int n = matrix.GetLength(0); // convenience
            float[,] result = Duplicate(matrix, sizeof(float));
            perm = new int[n];
            for (int i = 0; i < n; ++i)
            {
                perm[i] = i;
            }
            toggle = 1;
            for (int j = 0; j < n - 1; ++j) // each column
            {
                float colMax = Math.Abs(result[j, j]); // largest val in col j
                int pRow = j;
                for (int i = j + 1; i < n; ++i)
                {
                    if (result[i, j] > colMax)
                    {
                        colMax = result[i, j];
                        pRow = i;
                    }
                }
                if (pRow != j) // swap rows
                {
                    SwapRows(matrix, pRow, j);
                    int tmp = perm[pRow]; // and swap perm info
                    perm[pRow] = perm[j];
                    perm[j] = tmp;
                    toggle = -toggle; // row-swap toggle
                }
                if (Math.Abs(result[j, j]) < 1.0E-20)
                    return null; // consider a throw
                for (int i = j + 1; i < n; ++i)
                {
                    result[i, j] /= result[j, j];
                    for (int k = j + 1; k < n; ++k)
                        result[i, k] -= result[i, j] * result[j, k];
                }
            } // main j column loop
            return result;
        }




        private static void SwapRows<T>(T[,] matrix, int idxa, int idxb)
        {
            Parallel.For(0, matrix.GetLength(1), x =>
            {
                T d = matrix[idxa, x];
                matrix[idxa, x] = matrix[idxb, x];
                matrix[idxb, x] = d;
            });
        }

        public static T[,] Duplicate<T>(T[,] matrix, int sizeOfT)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var clone = new T[rows, cols];

            Buffer.BlockCopy(matrix, 0, clone, 0, sizeOfT * matrix.Length);

            return clone;
        }

        private static double[] HelperSolve(double[,] luMatrix, double[] b)
        {
            // solve luMatrix * x = b
            int n = luMatrix.GetLength(0);
            var x = new double[n];
            b.CopyTo(x, 0);

            for (int i = 1; i < n; ++i)
            {
                double sum = x[i];
                for (int j = 0; j < i; ++j)
                    sum -= luMatrix[i, j] * x[j];
                x[i] = sum;
            }
            x[n - 1] /= luMatrix[n - 1, n - 1];
            for (int i = n - 2; i >= 0; --i)
            {
                double sum = x[i];
                for (int j = i + 1; j < n; ++j)
                    sum -= luMatrix[i, j] * x[j];
                x[i] = sum / luMatrix[i, i];
            }
            return x;
        }

        private static float[] HelperSolve(float[,] luMatrix, float[] b)
        {
            // solve luMatrix * x = b
            int n = luMatrix.GetLength(0);
            var x = new float[n];
            b.CopyTo(x, 0);

            for (int i = 1; i < n; ++i)
            {
                float sum = x[i];
                for (int j = 0; j < i; ++j)
                    sum -= luMatrix[i, j] * x[j];
                x[i] = sum;
            }
            x[n - 1] /= luMatrix[n - 1, n - 1];
            for (int i = n - 2; i >= 0; --i)
            {
                float sum = x[i];
                for (int j = i + 1; j < n; ++j)
                    sum -= luMatrix[i, j] * x[j];
                x[i] = sum / luMatrix[i, i];
            }
            return x;
        }



        public static double[,] Inverse(double[,] matrix)
        {
            int n = matrix.GetLength(0);
            double[,] result = Duplicate(matrix, sizeof(double));
            int[] perm;
            int toggle;
            double[,] lum = Decompose(matrix, out perm, out toggle);
            if (lum == null)
                throw new MatrixException("Unable to compute inverse");
            var b = new double[n];
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == perm[j])
                        b[j] = 1.0;
                    else
                        b[j] = 0.0;
                }
                double[] x = HelperSolve(lum, b);
                for (int j = 0; j < n; ++j)
                    result[j, i] = x[j];
            }
            return result;
        }

        public static float[,] Inverse(float[,] matrix)
        {
            int n = matrix.GetLength(0);
            float[,] result = Duplicate(matrix, sizeof(float));
            int[] perm;
            int toggle;
            float[,] lum = Decompose(matrix, out perm, out toggle);
            if (lum == null)
                throw new MatrixException("Unable to compute inverse");
            var b = new float[n];
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (i == perm[j])
                        b[j] = 1.0f;
                    else
                        b[j] = 0.0f;
                }
                float[] x = HelperSolve(lum, b);
                for (int j = 0; j < n; ++j)
                    result[j, i] = x[j];
            }
            return result;
        }




        public static double Determinant(double[,] matrix)
        {
            int[] perm;
            int toggle;
            double[,] lum = Decompose(matrix, out perm, out toggle);
            if (lum == null)
                throw new Exception("Unable to compute MatrixDeterminant");
            double result = toggle;
            for (int i = 0; i < lum.Length; ++i)
                result *= lum[i, i];
            return result;
        }

        public static float Determinant(float[,] matrix)
        {
            int[] perm;
            int toggle;
            float[,] lum = Decompose(matrix, out perm, out toggle);
            if (lum == null)
                throw new Exception("Unable to compute MatrixDeterminant");
            float result = toggle;
            for (int i = 0; i < lum.Length; ++i)
                result *= lum[i, i];
            return result;
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

        public static unsafe void Fill(double[,] matrix, double value)
        {
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(matrix, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();
                int width = matrix.GetLength(1);

                Parallel.For(0, matrix.GetLength(0),
                    i => Parallel.For(0, matrix.GetLength(1), j =>
                    {
                        UnsafeUpdate2DArray(arr, width, i, j, value);
                        //matrix[i, j] = value;
                    }));
            }
            finally
            {
                handle.Free();
            }
        }

        public static void Fill(float[,] matrix, float value)
        {
            Parallel.For(0, matrix.GetLength(0),
                i => Parallel.For(0, matrix.GetLength(1), j => { matrix[i, j] = value; }));
        }

        public static float[,] OnesF(int rows, int cols)
        {
            var arr = new float[rows, cols];
            Fill(arr, 1f);
            return arr;
        }

        public static float[,] ZerosF(int rows, int cols)
        {
            var arr = new float[rows, cols];
            Fill(arr, 0f);
            return arr;
        }

        public static double[,] OnesD(int rows, int cols)
        {
            var arr = new double[rows, cols];
            Fill(arr, 1d);
            return arr;
        }

        public static double[,] ZerosD(int rows, int cols)
        {
            var arr = new double[rows, cols];
            Fill(arr, 0d);
            return arr;
        }

        public static unsafe void InsertValuesFrom(double[,] target, int mPos, int nPos, double[,] src, int mSize = 0,
            int nSize = 0)
        {
            mSize = mSize == 0 ? target.GetLength(0) - mPos : mSize;
            nSize = nSize == 0 ? target.GetLength(1) - nPos : nSize;
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(target, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();
                int width = target.GetLength(1);

                Parallel.For(0, mSize, i =>
                    Parallel.For(0, nSize, j =>
                    {
                        UnsafeUpdate2DArray(arr, width, i + mPos, j + nPos, src[i, j]);
                        //target[i + mPos, j + nPos] = src[i, j];
                    }));
            }
            finally
            {
                handle.Free();
            }
        }

        public static unsafe double[,] GreaterThan(double[,] m1, double[,] m2)
        {
            int len = m1.GetLength(0), width = m1.GetLength(1);

            var result = new double[len, width];
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(result, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();

                Parallel.For(0, result.GetLength(0), i => Parallel.For(0, result.GetLength(1), j =>
                {
                    UnsafeUpdate2DArray(arr, width, i, j, Convert.ToDouble(m1[i, j] > m2[i, j]));
                    //result[i, j] = Convert.ToDouble(m1[i, j] > m2[i, j]);
                }));
            }
            finally
            {
                handle.Free();
            }


            return result;
        }

        public static double[,] LessThan(double[,] m1, double[,] m2)
        {
            return GreaterThan(m2, m1);
        }

        public static float[,] GreaterThan(float[,] m1, float[,] m2)
        {
            var result = new float[m1.GetLength(0), m1.GetLength(1)];

            Parallel.For(0, result.GetLength(0),
                i =>
                    Parallel.For(0, result.GetLength(1), j => { result[i, j] = Convert.ToSingle(m1[i, j] > m2[i, j]); }));
            return result;
        }

        public static float[,] LessThan(float[,] m1, float[,] m2)
        {
            return GreaterThan(m2, m1);
        }

        internal static unsafe double[,] SubMatrix(double[,] matrix, int startRow, int startCol, int numRows = 0,
            int numCols = 0)
        {
            numRows = numRows != 0 ? numRows : matrix.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : matrix.GetLength(1) - startCol;

            var result = new double[numRows, numCols];
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(result, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();

                Parallel.For(0, numRows, i => Parallel.For(0, numCols, j =>
                {
                    UnsafeUpdate2DArray(arr, numCols, i, j, matrix[i + startRow, j + startCol]);
                    //result[i, j] = matrix[i + startRow, j + startCol];
                }));
            }
            finally
            {
                handle.Free();
            }


            return result;
        }

        internal static double[,] ToVector(double[,] matrix)
        {
            if (matrix.GetLength(1) == 1)
            {
                return SubMatrix(matrix, 0, 0, 0, 1);
            }
            if (matrix.GetLength(0) == 1)
            {
                return SubMatrix(matrix, 0, 0, 1);
            }

            throw new MatrixException("Matrix is not a one liner");
        }

        internal static unsafe void InsertValuesFromRowOrColumn(double[,] matrix, double[,] src, int length = 0,
            bool fromColumn = true,
            int mPos = 0, int nPos = 0)
        {
            length = length == 0 ? Math.Max(src.GetLength(0), src.GetLength(1)) : length;
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(matrix, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();
                int width = matrix.GetLength(1);

                Parallel.For(0, length, i =>
                {
                    if (fromColumn)
                    {
                        UnsafeUpdate2DArray(arr, width, mPos + i, nPos, src[i, 0]);
                        //matrix[mPos + i, nPos] = src[i, 0];
                    }
                    else
                    {
                        UnsafeUpdate2DArray(arr, width, mPos, nPos + i, src[0, i]);
                        //matrix[mPos, nPos + i] = src[0, i];
                    }
                });
            }
            finally
            {
                handle.Free();
            }
        }

        public static unsafe void UpdateValueAlongAxis(double[,] matrix, int index, int value, Axis axis)
        {
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(matrix, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();
                int width = matrix.GetLength(1);

                if (axis == Axis.Horizontal)
                {
                    Parallel.For(0, matrix.GetLength(1), i =>
                    {
                        UnsafeUpdate2DArray(arr, width, index, i, value);
                        //matrix[index, i] = value;
                    });
                }
                else
                {
                    Parallel.For(0, matrix.GetLength(0), i =>
                    {
                        UnsafeUpdate2DArray(arr, width, i, index, value);
                        //matrix[i, index] = value;
                    });
                }
            }
            finally
            {
                handle.Free();
            }
        }

        public static unsafe double[,] Add(double[,] m1, double[,] m2)
        {
            var res = new double[m1.GetLength(0), m1.GetLength(1)];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(res, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();
                int width = res.GetLength(1);


                Parallel.For(0, res.GetLength(0), i => Parallel.For(0, res.GetLength(1), j =>
                {
                    UnsafeUpdate2DArray(arr, width, i, j, m1[i, j] + m2[i, j]);
                    //res[i, j] = m1[i, j] + m2[i, j];
                }));
                return res;
            }
            finally
            {
                handle.Free();
            }
        }

        public static unsafe double[,] Subtract(double[,] m1, double[,] m2)
        {
            var res = new double[m1.GetLength(0), m1.GetLength(1)];
            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(res, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();
                int width = res.GetLength(1);

                Parallel.For(0, res.GetLength(0), i => Parallel.For(0, res.GetLength(1), j =>
                {
                    UnsafeUpdate2DArray(arr, width, i, j, m1[i, j] - m2[i, j]);
                    //res[i, j] = m1[i, j] - m2[i, j];
                }));
                return res;
            }
            finally
            {
                handle.Free();
            }
        }

        public static unsafe double[,] Pow(double[,] matrix, double power)
        {
            var res = new double[matrix.GetLength(0), matrix.GetLength(1)];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(res, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();
                int width = res.GetLength(1);

                if (power == 2d)
                {
                    Parallel.For(0, res.GetLength(0), i => Parallel.For(0, res.GetLength(1), j =>
                    {
                        UnsafeUpdate2DArray(arr, width, i, j, matrix[i, j] * matrix[i, j]);
                        //res[i, j] = matrix[i, j]*matrix[i, j];
                    }));
                }
                else
                {
                    Parallel.For(0, res.GetLength(0), i => Parallel.For(0, res.GetLength(1), j =>
                    {
                        UnsafeUpdate2DArray(arr, width, i, j, Math.Pow(matrix[i, j], power));
                        //res[i, j] = Math.Pow(matrix[i, j], power);
                    }));
                }
                return res;
            }
            finally
            {
                handle.Free();
            }
        }

        public static IEnumerable<T> EnumerateElements<T>(T[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    yield return matrix[i, j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void UnsafeUpdate2DArray(double* array, int dim1Length, int row, int col, double value)
        {
            array[dim1Length * row + col] = value;
        }
    }
}