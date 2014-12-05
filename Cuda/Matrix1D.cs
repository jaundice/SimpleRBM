using System;
using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public class Matrix1D<T> : Matrix
    {
        public Matrix1D(GPGPU gpu, Array array, int[] dimensions)
            : base(gpu, array, dimensions)
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

        public static implicit operator T[](Matrix1D<T> m)
        {
            return m.Matrix;
        }
    }
}