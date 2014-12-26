using Cudafy.Host;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public static class GPGPUEx
    {
        public static Matrix1D<T> AllocateAndSet<T>(this GPGPU gpu, int rows) where T : struct
        {
            T[] res = gpu.Allocate<T>(rows);
            gpu.Set(res);
            return new Matrix1D<T>(gpu, res, rows);
        }

        public static Matrix2D<T> AllocateAndSet<T>(this GPGPU gpu, int rows, int cols) where T : struct
        {
            T[,] res = gpu.Allocate<T>(rows, cols);
            gpu.Set(res);
            return new Matrix2D<T>(gpu, res, new[] { rows, cols });
        }

        //public static void Free<T>(this GPGPU self, Matrix m) //temp hack
        //{
        //    m.Dispose();
        //}
    }
}