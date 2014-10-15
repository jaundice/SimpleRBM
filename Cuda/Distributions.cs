using System;
using Cudafy;
using Cudafy.Maths.RAND;
using Cudafy.Rand;

namespace CudaRbm
{
    //public class Distributions
    //{
    //    [Cudafy]
    //    public static void UniformRandomMatrix(GThread thread, RandStateXORWOW[] randState, double[,] target)
    //    {
    //        int i = thread.blockIdx.x + thread.blockIdx.x * thread.blockDim.x;
    //        int j = thread.blockIdx.y + thread.blockIdx.y * thread.blockDim.y;


    //        while (i < target.GetLength(0))
    //        {
    //            while (j < target.GetLength(1))
    //            {
    //                target[i, j] = thread.curand_uniform_double(ref randState[i]);

    //                j += thread.blockIdx.y * thread.blockDim.y;
    //            }
    //            i += thread.blockIdx.x * thread.blockDim.x;
    //        }
    //    }

    //    [Cudafy]
    //    public static void GaussianMatrix(GThread thread, RandStateXORWOW[] randState1, double[,] target)
    //    {
    //        int i = thread.blockIdx.x + thread.blockIdx.x * thread.blockDim.x;
    //        int j = thread.blockIdx.y + thread.blockIdx.y * thread.blockDim.y;


    //        while (i < target.GetLength(0))
    //        {
    //            while (j < target.GetLength(1))
    //            {
    //                //target[i, j] = Math.Sqrt(-2.0*Math.Log(thread.c))*
    //                //               Math.Sin(2.0*Math.PI*randState2[i, j].boxmuller_extra_double);

    //                target[i, j] = thread.curand_normal_double(ref randState1[i]);

    //                j += thread.blockIdx.y * thread.blockDim.y;
    //            }
    //            i += thread.blockIdx.x * thread.blockDim.x;
    //        }
    //    }


    //    [Cudafy]
    //    public static void UniformRandromVector(GThread thread, RandStateXORWOW[] randState, double[,] target)
    //    {
    //        int i = thread.blockIdx.x + thread.blockIdx.x * thread.blockDim.x;
    //        int j = thread.blockIdx.y + thread.blockIdx.y * thread.blockDim.y;

    //        while (i < target.GetLength(0))
    //        {
    //            while (j < 1)
    //            {
    //                target[i, j] =  thread.curand_uniform_double(ref randState[i]);;

    //                j += thread.blockIdx.y * thread.blockDim.y;
    //            }
    //            i += thread.blockIdx.x * thread.blockDim.x;
    //        }
    //    }
    //}
}