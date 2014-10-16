using System;
using System.Linq;
using Cudafy.Host;

namespace CudaRbm
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
            Dispose(true);
        }

        public int GetLength(int dimension)
        {
            return _dimensions[dimension];
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (Location == Location.GPU)
                {
                    GPU.Free(InnerMatrix);
                }
            }
        }
    }
}