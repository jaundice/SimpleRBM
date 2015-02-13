using System;
using System.Threading.Tasks;
using Cudafy;

namespace SimpleRBM.Cuda
{
    public static partial class Matrix2DCuda
    {
        public const uint TRUE = 1u;
        public const uint FALSE = 0u;

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