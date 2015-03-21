using System;
using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public class Matrix1D<T> : Matrix where T : struct
    {
        public Matrix1D(GPGPU gpu, Array array, int rows)
            : base(gpu, array, new[] { rows })
        {
        }

        public Matrix1D(Array array)
            : base(array)
        {
        }

        public T[] Matrix
        {
            get { return (T[])base.InnerMatrix; }
        }

        public Matrix2D<T> Cast2D(int rows, int cols)
        {
            if (rows * cols > GetLength(0))
                throw new Exception("Invalid dimensions");
            return new Matrix2D<T>(GPU, GPU.Cast(Matrix, rows, cols), new[] { rows, cols });
        }

    }
}