using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public static class GPGPUEx
    {
        public static Matrix1D<T> AllocateAndSet<T>(this GPGPU gpu, int rows) where T : struct
        {
            Matrix1D<T> res = gpu.AllocateNoSet<T>(rows);
            //res.Set();
            return res;
        }

        public static Matrix2D<T> AllocateAndSet<T>(this GPGPU gpu, int rows, int cols) where T : struct
        {
            Matrix2D<T> res = gpu.AllocateNoSet<T>(rows, cols);
            //res.Set();
            return res;
        }

        public static Matrix1D<T> AllocateNoSet<T>(this GPGPU gpu, int rows) where T : struct
        {
            T[] res = gpu.Allocate<T>(rows);
            gpu.Set(res);
            return new Matrix1D<T>(gpu, res, rows);
        }

        public static Matrix2D<T> AllocateNoSet<T>(this GPGPU gpu, int rows, int cols) where T : struct
        {
            T[,] res = gpu.Allocate<T>(rows, cols);
            gpu.Set(res);
            return new Matrix2D<T>(gpu, res, new[] { rows, cols });
        }

        public static Matrix2D<T> Upload<T>(this GPGPU gpu, T[,] source) where T : struct
        {
            Matrix2D<T> tempSrcData = gpu.AllocateNoSet<T>(source.GetLength(0), source.GetLength(1));
            gpu.CopyToDevice(source, tempSrcData.Matrix);
            return tempSrcData;
        }

        public static Matrix1D<T> Upload<T>(this GPGPU gpu, T[] source) where T : struct
        {
            Matrix1D<T> tempSrcData = gpu.AllocateNoSet<T>(source.GetLength(0));
            gpu.CopyToDevice(source, tempSrcData.Matrix);
            return tempSrcData;
        }

        //public static Matrix2D<T> UploadToConstantMemory<T>(this GPGPU gpu, T[,] source) where T : struct
        //{
        //    var res = gpu.Allocate<T>(source.GetLength(0), source.GetLength(1));
        //    gpu.CopyToConstantMemory(source, res);
        //    return new Matrix2D<T>(gpu, res, new[] { source.GetLength(0), source.GetLength(1) });
        //}
    }
}