using System;
using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public class Matrix2D<T> : Matrix where T : struct
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

        public Matrix1D<T> Cast1D()
        {
            return new Matrix1D<T>(this.GPU, this.GPU.Cast<T>(Matrix, GetLength(0) * GetLength(1)), GetLength(0) * GetLength(1));
        }

        public Matrix2D<T> CloneOnDevice()
        {
            var clone = GPU.AllocateNoSet<T>(this.GetLength(0), this.GetLength(1));
            using(var selfAs1d = this.Cast1D())
            using (var clone1d = clone.Cast1D())
            {
                GPU.CopyOnDevice(selfAs1d.Matrix, clone1d.Matrix);
            }
            return clone;
        } 
    }
}