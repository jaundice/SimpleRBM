using System;
using System.Diagnostics;
using Cudafy;

namespace SimpleRBM.Cuda
{
    public class ThreadOptimiser
    {
        public static ThreadOptimiser Instance;

        public ThreadOptimiser(Version cudaVersion, int multiprocessors, int maxThreadsPerBlock,
            int maxThreadsPerMultiProcessor, dim3 maxGridSize,
            dim3 maxBlockSize)
        {
            CudaVersion = cudaVersion;
            MultiProcessorCount = multiprocessors;
            MaxThreadsPerBlock = maxThreadsPerBlock;
            MaxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor;
            MaxGridSize = maxGridSize;
            MaxBlockSize = maxBlockSize;
        }

        public Version CudaVersion { get; protected set; }
        public dim3 MaxGridSize { get; protected set; }

        public int MultiProcessorCount { get; protected set; }
        public int MaxThreadsPerBlock { get; protected set; }

        public int MaxThreadsPerMultiProcessor { get; protected set; }
        public dim3 MaxBlockSize { get; protected set; }

        public int MaxResidentBlocksPerProcessor
        {
            get
            {
                if (CudaVersion.Major < 3)
                {
                    return 8;
                }
                if (CudaVersion.Major < 5)
                {
                    return 16;
                }
                return 32;
            }
        }

        public int MaxResidentWarpsPerProcessor
        {
            get
            {
                if (CudaVersion < new Version(1, 2))
                {
                    return 24;
                }
                if (CudaVersion < new Version(2, 0))
                {
                    return 32;
                }
                if (CudaVersion < new Version(3, 0))
                {
                    return 48;
                }
                return 64;
            }
        }

        public int WarpSize
        {
            get { return 32; }
        }

        public void GetStrategy(Matrix m, out dim3 grid, out dim3 block)
        {
            GetStrategy(m.GetLength(0), m.GetLength(1), out grid, out block);
        }

        public void GetStrategy(int rows, int cols, out dim3 grid, out dim3 block)
        {
            int big = Math.Max(rows, cols);
            int small = Math.Min(rows, cols);


            int sm = small == 1 ? 1 : 16;
            int bi = small == 1 ? 32 : 16;

            if (big == small)
            {
                var d = big < 32 ? 16 : 32;

                block = new dim3(d, d);
                var g = Math.Max(1, (int)Math.Round((double)big / d));
                grid = new dim3(g, g);

            }
            else if (big <= 1024 && small > 1)
            {
                var rat = (double)small / big;
                dim3 b, g;

                if (rat >= 0.5)
                {

                    var d = big < 32 ? 16 : 32;
                    b = new dim3(d, d);
                    g = new dim3(Math.Max((int)Math.Round(big / (double)d), 1), Math.Max((int)Math.Round(small / (double)d), 1));
                }
                else
                {
                    b = new dim3((int)Math.Ceiling(big / 32.0) * 32, 1);
                    g = new dim3(1, small);

                }
                if (rows > cols)
                {
                    block = b;
                    grid = g;
                }
                else
                {
                    block = new dim3(b.y, b.x);
                    grid = new dim3(g.y, g.x);
                }
            }
            else
            {
                if (small >= 1024)
                {
                    bi = 1;
                    sm = 1024;
                }
                else if (small >= 672)
                {
                    bi = 1;
                    sm = 672;
                }
                else if (small >= 512)
                {
                    bi = 1;
                    sm = 512;
                }
                else if (small >= 256)
                {
                    bi = 1;
                    sm = 256;
                }
                else if (small >= 32)
                {
                    bi = 1;
                    sm = (int)Math.Round(small / 32.0, MidpointRounding.ToEven) * 32;
                }

                var b = new dim3(bi, sm);
                var g = new dim3((int)Math.Max(1, Math.Floor((double)big / bi)), Math.Max(1, (int)Math.Floor((double)small / sm)));
                if (rows > cols)
                {
                    block = b;
                    grid = g;
                }
                else
                {
                    block = new dim3(b.y, b.x);
                    grid = new dim3(g.y, g.x);
                }

            }





#if DEBUG
            Trace.TraceInformation("Generating Strategy for {{ rows:{0}, cols:{1} }}", rows, cols);
            Trace.TraceInformation("block {{ x:{0}, y:{1} }} grid {{ x:{2}, y:{3} }}", block.x, block.y, grid.x, grid.y);
#endif
        }

    }
}