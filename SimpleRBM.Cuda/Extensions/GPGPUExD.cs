using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Cuda.CudaMatrix;
using TElement = System.Double;
namespace SimpleRBM.Cuda
{
    public static class GPGPUExD
    {
        public static Matrix2D<TElement> GuassianDistribution(this GPGPU gpu, GPGPURAND rand, int x, int y, TElement mean = 0,
            TElement stDev = 0.5, TElement scale = 1.0)
        {

            var len = x * y;
            if (len % 2 != 0)
                len++;
            var ret = gpu.AllocateNoSet<TElement>(x, y);
            using (var tempGaussian = ret.Cast1D())
            {
                rand.GenerateNormal(tempGaussian.Matrix, (float)mean, (float)stDev/*, len*/);
            }
            if (scale != 1.0)
                ret.Multiply(scale);
            return ret;
        }


        public static Matrix2D<TElement> UniformDistribution(this GPGPU gpu, GPGPURAND rand, int x, int y,
            TElement scale = 1.0)
        {
            Matrix2D<TElement> array = gpu.AllocateNoSet<TElement>(x, y);
            using (Matrix1D<TElement> tempUniform = array.Cast1D())
            {
                rand.GenerateUniform(tempUniform.Matrix, x * y);
            }
            if (scale != 1.0)
                array.MultiplyInPlace(scale);

            return array;
        }

        public static void UniformDistributionBool(this GPGPU gpu, GPGPURAND rand, int x, int y, out Matrix2D<TElement> result)
        {
            Matrix2D<TElement> array = UniformDistribution(gpu, rand, x, y);
            dim3 grid, block;

            ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);
            gpu.Launch(grid, block, Matrix2DCuda.ToBinaryD, array.Matrix);

            result = array;
        }
    }
}