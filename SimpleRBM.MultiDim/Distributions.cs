using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace SimpleRBM.MultiDim
{
    public static class Distributions
    {
        //[ThreadStatic]
        //private static Random _random;
        ////reuse this if you are generating many non time dependant numbers

        //public static Random Random
        //{
        //    get { return _random ?? (_random = new Random()); }
        //}



        private static Random Random = new Random();
        //reuse this if you are generating many non time dependant numbers

        //public static Random Random
        //{
        //    get { return _random; }
        //}



        /// <summary>
        ///     Random is not a thread-safe class, this helper function locks our global Random instance.
        /// </summary>
        /// <returns></returns>
        public static double GetRandomDouble()
        {
            lock (Random)
                return Random.NextDouble();
        }


        /// <summary>
        ///     u(0,1) normal dist
        /// </summary>
        public static double GaussianNormal()
        {
            double u1 = 0;
            while (u1 == 0.0)
                u1 = GetRandomDouble();
            double u2 = GetRandomDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return randStdNormal;
        }



        /// <summary>
        ///     Random Gaussian Matrix
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        /// <returns></returns>
        public unsafe static double[,] GaussianMatrixD(int rows, int cols)
        {
            //var matrix = new double[rows, cols];

            //Parallel.For(0, rows, i => Parallel.For(0, cols, j => { matrix[i, j] = GaussianNormal(); }));

            //return matrix;

            var matrix = new double[rows, cols];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(matrix, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();


                ThreadUtil.Run(rows, cols, (i, j) => Matrix2D.UnsafeUpdate2DArray(arr, cols, i, j, GaussianNormal()));


                return matrix;
            }
            finally
            {
                handle.Free();
            }
        }

        public unsafe static float[,] GaussianMatrixF(int rows, int cols)
        {
            //var matrix = new double[rows, cols];

            //Parallel.For(0, rows, i => Parallel.For(0, cols, j => { matrix[i, j] = GaussianNormal(); }));

            //return matrix;

            var matrix = new float[rows, cols];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(matrix, GCHandleType.Pinned);
                var arr = (float*)handle.AddrOfPinnedObject();


                ThreadUtil.Run(rows, cols, (i, j) => Matrix2D.UnsafeUpdate2DArray(arr, cols, i, j, (float)GaussianNormal()));


                return matrix;
            }
            finally
            {
                handle.Free();
            }
        }


        /// <summary>
        ///     Uniform Random Matrix
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        /// <returns></returns>
        public unsafe static double[,] UniformRandromMatrixD(int rows, int cols)
        {
            var matrix = new double[rows, cols];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(matrix, GCHandleType.Pinned);
                var arr = (double*)handle.AddrOfPinnedObject();


                ThreadUtil.Run(rows, cols, (i, j) =>  Matrix2D.UnsafeUpdate2DArray(arr, cols, i, j, GetRandomDouble()));


                return matrix;
            }
            finally
            {
                handle.Free();
            }
        }


        public unsafe static float[,] UniformRandromMatrixF(int rows, int cols)
        {
            var matrix = new float[rows, cols];

            GCHandle handle = default(GCHandle);
            try
            {
                handle = GCHandle.Alloc(matrix, GCHandleType.Pinned);
                var arr = (float*)handle.AddrOfPinnedObject();


                ThreadUtil.Run(rows, cols, (i, j) =>  Matrix2D.UnsafeUpdate2DArray(arr, cols, i, j, (float)GetRandomDouble()));


                return matrix;
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        ///     Uniform Random Boolean Matrix
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        /// <returns></returns>
        public static double[,] UniformRandromMatrixBoolD(int rows, int cols)
        {
            var matrix = new double[rows, cols];

            ThreadUtil.Run(rows, cols, (i, j) =>  matrix[i, j] = Convert.ToInt32(GetRandomDouble()));

            return matrix;
        }

        public static float[,] UniformRandromMatrixBoolF(int rows, int cols)
        {
            var matrix = new float[rows, cols];

            ThreadUtil.Run(rows, cols, (i, j) =>  matrix[i, j] = Convert.ToInt32(GetRandomDouble()));

            return matrix;
        }

        /// <summary>
        ///     Uniform Random Vector
        /// </summary>
        /// <param name="numElements"></param>
        /// <returns></returns>
        public static double[,] UniformRandromVectorD(int numElements)
        {
            var vector = new double[1, numElements];

            Parallel.For(0, numElements, i => { vector[0, i] = GetRandomDouble(); });
            return vector;
        }

        public static float[,] UniformRandromVectorF(int numElements)
        {
            var vector = new float[1, numElements];

            Parallel.For(0, numElements, i => { vector[0, i] = (float)GetRandomDouble(); });
            return vector;
        }
    }
}