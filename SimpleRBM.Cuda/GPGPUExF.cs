using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using TElement = System.Single;
namespace SimpleRBM.Cuda
{
    public static class GPGPUExF
    {
        public static Matrix2D<TElement> GuassianDistribution(this GPGPU gpu, GPGPURAND rand, int x, int y, TElement mean = 0f,
            TElement stDev = 0.5f, TElement scale = 1.0f)
        {
            Matrix2D<TElement> array = gpu.AllocateNoSet<TElement>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);

            using (Matrix1D<TElement> tempGaussian = gpu.AllocateNoSet<TElement>(x * y))
            {
                int len = x * y;
                if (len % 2 != 0)
                    len++;

                rand.GenerateNormal(tempGaussian, mean, stDev, len);
                gpu.Launch(grid, block, Matrix2DCuda.CopyToArrayAtNF2, array.Matrix, tempGaussian.Matrix, scale);
            }
            return array;
        }


        public static Matrix2D<TElement> UniformDistribution(this GPGPU gpu, GPGPURAND rand, int x, int y, TElement scale = 1.0f)
        {
            Matrix2D<TElement> array = gpu.AllocateNoSet<TElement>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);

            using (Matrix1D<TElement> tempUniform = gpu.AllocateNoSet<TElement>(x * y))
            {
                rand.GenerateUniform(tempUniform, x * y);

                gpu.Launch(grid, block, Matrix2DCuda.CopyToArrayAtNF2, array.Matrix, tempUniform.Matrix, scale);
            }
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