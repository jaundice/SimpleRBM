using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy.Host;
using Cudafy.Maths.BLAS;

namespace SimpleRBM.Cuda.CudaMatrix
{
    public class CudaToolsRegistry
    {
        private static ConcurrentDictionary<CudaGPU, GPGPUBLAS> _storage = new ConcurrentDictionary<CudaGPU, GPGPUBLAS>();

        public static GPGPUBLAS GetBlas(GPGPU gpu)
        {
            return _storage.GetOrAdd((CudaGPU)gpu, GPGPUBLAS.Create);
        }

        public static bool RemoveBlas(GPGPU gpu, out GPGPUBLAS blas)
        {
            return _storage.TryRemove((CudaGPU)gpu, out blas);
        }
    }
}
