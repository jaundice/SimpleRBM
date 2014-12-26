using System;
using System.Diagnostics;
using System.Linq;
using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public class Matrix : IDisposable
    {
        protected readonly int[] _dimensions;

        protected Array InnerMatrix;

        public Matrix(GPGPU gpu, Array array, int[] dimensions)
        {
            Location = Location.GPU;
            _dimensions = dimensions;
            GPU = gpu;
            InnerMatrix = array;
        }

        public Matrix(Array array)
        {
            Location = Location.MainMemory;
            _dimensions = Enumerable.Range(0, array.Rank).Select(array.GetLength).ToArray();
        }

        public Location Location { get; protected set; }

        protected internal GPGPU GPU { get; set; }


        public void Dispose()
        {
            if (!Disposed)
            {
                Disposed = true;
                Dispose(true);
                GC.SuppressFinalize(this);
            }
        }

        public bool Disposed { get; protected set; }

        public int GetLength(int dimension)
        {
            return _dimensions[dimension];
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (Location == Location.GPU)
                {
                    GPU.Free(InnerMatrix);
                }
            }
        }

        ~Matrix()
        {
            Trace.TraceError("Matrix finalizer called. Dispose properly!");
            Dispose(false);
        }
    }
}