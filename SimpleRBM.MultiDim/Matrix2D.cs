using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SimpleRBM.MultiDim
{
    public static partial class Matrix2D
    {
        public enum Axis
        {
            Horizontal = 0,
            Vertical = 1
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

        public static T[,] JaggedToMultidimesional<T>(T[][] source)
        {
            int rows = source.GetLength(0);
            int cols = source[0].GetLength(0);

            var res = new T[rows, cols];
            ThreadUtil.Run(rows, cols, (i, j) => res[i, j] = source[i][j]);

            return res;
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
    }
}