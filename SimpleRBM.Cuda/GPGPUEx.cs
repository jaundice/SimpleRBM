using Cudafy.Host;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public static class GPGPUEx
    {
        public static Matrix1D<T> AllocateAndSet<T>(this GPGPU gpu, int rows) where T : struct
        {
            Matrix1D<T> res = gpu.AllocateNoSet<T>(rows);
            res.Set();
            return res;
        }

        public static Matrix2D<T> AllocateAndSet<T>(this GPGPU gpu, int rows, int cols) where T : struct
        {
            Matrix2D<T> res = gpu.AllocateNoSet<T>(rows, cols);
            res.Set();
            return res;
        }

        public static Matrix1D<T> AllocateNoSet<T>(this GPGPU gpu, int rows) where T : struct
        {
            T[] res = gpu.Allocate<T>(rows);
            return new Matrix1D<T>(gpu, res, rows);
        }

        public static Matrix2D<T> AllocateNoSet<T>(this GPGPU gpu, int rows, int cols) where T : struct
        {
            T[,] res = gpu.Allocate<T>(rows, cols);
            return new Matrix2D<T>(gpu, res, new[] { rows, cols });
        }

    }
}