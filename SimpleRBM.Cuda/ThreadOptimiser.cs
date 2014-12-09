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
            if (cols > 1)
            {
                int x = rows > 1 ? 32 : 1;
                int y = 32;
                if (cols > 1024)
                {
                    x = 1;
                    y = 1024;
                }
                else if (cols > 512)
                {
                    x = 1;
                    y = 512;
                }
                else if (cols > 256)
                {
                    x = 1;
                    y = 256;
                }
                block = new dim3(x, y);
                grid = new dim3(Math.Max(1, (int)Math.Floor((double)rows / x)),
                    x == 1 ? (int)Math.Max(1, Math.Floor((double)cols / y)) : 1);
            }
            else
            {
                int y = cols > 1 ? 32 : 1;
                int x = 32;
                if (rows > 1024)
                {
                    y = 1;
                    x = 1024;
                }
                else if (rows > 512)
                {
                    y = 1;
                    x = 512;
                }
                else if (rows > 256)
                {
                    y = 1;
                    x = 256;
                }
                block = new dim3(x, y);
                grid = new dim3(y == 1 ? Math.Max(1, (int)Math.Floor((double)rows / x)) : 1,
                    Math.Max(1, (int)Math.Floor((double)cols / y)));
            }

            Trace.TraceInformation("Generating Strategy for {{ rows:{0}, cols:{1} }}", rows, cols);
            Trace.TraceInformation("block {{ x:{0}, y:{1} }} grid {{ x:{2}, y:{3} }}", block.x, block.y, grid.x, grid.y);
        }


        //public void GetStrategy(int rows, int cols, out dim3 grid, out dim3 block)
        //{
        //    var maxResidentThreads = MultiProcessorCount * MaxResidentWarpsPerProcessor * WarpSize;

        //    var warpsPerRow = Math.Max(1, (int)Math.Round((double)cols / WarpSize, MidpointRounding.AwayFromZero));

        //    var warpsPerBlock = MaxThreadsPerBlock / WarpSize;

        //    var blocksPerRow = warpsPerRow / warpsPerBlock;

        //    var numIterations = Math.Max(1, (int)Math.Round((double)(rows * cols) / maxResidentThreads, MidpointRounding.AwayFromZero));

        //    var threadsPerProc = ((double)rows * cols) / MultiProcessorCount;

        //    var warpsPerProc = Math.Max(1, (int)Math.Floor(threadsPerProc / WarpSize));

        //    var threadsPerBlock = Math.Max(1, (int)Math.Round((double)warpsPerProc / warpsPerBlock, MidpointRounding.AwayFromZero));

        //    var tpb = (int)Math.Max(1, Math.Floor((double)threadsPerBlock / WarpSize)) * WarpSize;

        //    if (tpb > cols)
        //    {

        //        block = new dim3(tpb, 1);
        //        grid = new dim3(1, Math.Max(1, (int)Math.Round((double)cols / MaxResidentBlocksPerProcessor, MidpointRounding.AwayFromZero)));
        //    }
        //    else
        //    {
        //        block = new dim3(1, tpb);
        //        grid = new dim3(Math.Max(1, (int)Math.Round((double)rows / MaxResidentBlocksPerProcessor, MidpointRounding.AwayFromZero)));
        //    }
        //    Trace.TraceInformation("Generating Strategy for {{ rows:{0}, cols:{1} }}", rows, cols);
        //    Trace.TraceInformation("block {{ x:{0}, y:{1} }} grid {{ x:{2}, y:{3} }}", block.x, block.y, grid.x, grid.y);

        //}

        //public void GetStrategy(int rows, int cols, out dim3 grid, out dim3 block)
        //{
        //    //temp for now
        //    //todo: work out better heuristic for assigning grids/blocks/threads

        //    //temp for now
        //    Trace.TraceInformation("Generating Strategy for {{ rows:{0}, cols:{1} }}", rows, cols);


        //    if (cols < 2)
        //    {
        //        grid = new dim3(Math.Max(1, (int)Math.Floor(rows / 512f)));
        //        block = new dim3(512, 1);
        //    }
        //    else if (rows < 2)
        //    {
        //        grid = new dim3(1, Math.Max(1, (int)Math.Floor(cols / 512f)));
        //        block = new dim3(1, 512);
        //    }
        //    else
        //    {
        //        //grid = new dim3(16);
        //        //block = new dim3(4, 256);
        //        if (rows >= cols)
        //        {
        //            int small, big;
        //            GetDimension(cols, out small, out big);
        //            grid = new dim3(GetGrid((int)Math.Floor(rows / (float)small)));
        //            block = new dim3(small, big);
        //        }
        //        else
        //        {
        //            int small, big;
        //            GetDimension(rows, out small, out big);
        //            grid = new dim3(1, GetGrid((int)Math.Floor(cols / (float)small)));
        //            block = new dim3(big, small);
        //        }
        //    }

        //    Trace.TraceInformation("block {{ x:{0}, y:{1} }} grid {{ x:{2}, y:{3} }}", block.x, block.y, grid.x, grid.y);
        //}

        private static int GetGrid(int input)
        {
            return Math.Max(1, (int) Math.Floor(input/2f));
        }

        private static void GetDimension(int width, out int small, out int big)
        {
            if (width > 256)
            {
                small = 2;
                big = 256;
                return;
            }
            if (width > 128)
            {
                small = 4;
                big = 128;
                return;
            }
            if (width > 64)
            {
                small = 8;
                big = 64;
                return;
            }

            small = 16;
            big = 32;
        }

        public class Strategy
        {
            public dim3 Grid { get; private set; }
            public dim3 Block { get; private set; }
        }
    }
}