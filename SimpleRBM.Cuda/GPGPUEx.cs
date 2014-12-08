using Cudafy.Host;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public static class GPGPUEx
    {
        public static Matrix1D<T> AllocateAndSet<T>(this GPGPU gpu, int x) where T : struct
        {
            T[] res = gpu.Allocate<T>(x);
            gpu.Set(res);
            return new Matrix1D<T>(gpu, res, new[] { x });
        }

        public static Matrix2D<T> AllocateAndSet<T>(this GPGPU gpu, int x, int y) where T : struct
        {
            T[,] res = gpu.Allocate<T>(x, y);
            gpu.Set(res);
            return new Matrix2D<T>(gpu, res, new[] { x, y });
        }

        public static void Free<T>(this GPGPU self, Matrix m) //temp hack
        {
            m.Dispose();
        }
    }
}