using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using TElement = System.Single;
namespace SimpleRBM.Cuda
{
    public static class GPGPUExF
    {
        public static Matrix2D<TElement> GuassianDistribution(this GPGPU gpu, GPGPURAND rand, int x, int y, TElement mean = 0,
            TElement stDev = 0.5f, TElement scale = 1.0f)
        {

            var ret = gpu.AllocateNoSet<TElement>(x, y);
            var len = x * y;
            if (len % 2 != 0)
            {
                len++;

                using (var tempGaussian = gpu.AllocateNoSet<TElement>(len))
                using (var ret1D = ret.Cast1D())
                {
                    rand.GenerateNormal(tempGaussian.Matrix, (float)mean, (float)stDev, len);
                    gpu.CopyOnDevice(tempGaussian.Matrix, 0, ret1D.Matrix, 0, len - 1);
                }
            }
            else
            {
                using (var tempGaussian = ret.Cast1D())
                {
                    rand.GenerateNormal(tempGaussian.Matrix, (float)mean, (float)stDev, len);
                }
            }
            if (scale != 1.0)
                ret.MultiplyInPlace(scale);
            return ret;
        }


        public static Matrix2D<TElement> UniformDistribution(this GPGPU gpu, GPGPURAND rand, int x, int y,
            TElement scale = 1.0f)
        {
            Matrix2D<TElement> array = gpu.AllocateNoSet<TElement>(x, y);
            using (Matrix1D<TElement> tempUniform = array.Cast1D())
            {
                rand.GenerateUniform(tempUniform.Matrix, x * y);
            }
            if (scale != 1.0f)
                array.MultiplyInPlace(scale);

            return array;
        }

        public static void UniformDistributionBool(this GPGPU gpu, GPGPURAND rand, int x, int y, out Matrix2D<TElement> result)
        {
            Matrix2D<TElement> array = UniformDistribution(gpu, rand, x, y);
            dim3 grid, block;

            ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);
            gpu.Launch(grid, block, Matrix2DCuda.ToBinaryF, array.Matrix);

            result = array;
        }
    }
}