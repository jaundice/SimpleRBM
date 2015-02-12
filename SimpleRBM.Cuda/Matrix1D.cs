using System;
using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public class Matrix1D<T> : Matrix
    {
        public Matrix1D(GPGPU gpu, Array array, int rows)
            : base(gpu, array, new[] {rows})
        {
        }

        public Matrix1D(Array array)
            : base(array)
        {
        }

        public T[] Matrix
        {
            get { return (T[]) base.InnerMatrix; }
        }

        public static implicit operator T[](Matrix1D<T> m)
        {
            return m.Matrix;
        }
    }
}