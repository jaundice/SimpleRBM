using System;
using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public class Matrix2D<T> : Matrix
    {
        public Matrix2D(GPGPU gpu, Array array, int[] dimensions)
            : base(gpu, array, dimensions)
        {
        }

        public Matrix2D(Array array)
            : base(array)
        {
        }

        public T[,] Matrix
        {
            get { return (T[,])base.InnerMatrix; }
        }

        public static implicit operator T[,](Matrix2D<T> m)
        {
            return m.Matrix;
        }
    }
}