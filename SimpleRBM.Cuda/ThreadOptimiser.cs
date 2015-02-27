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
            if (small >= 2048)
            {
                bi = 1;
                sm = 1024;
            }
            if (small >= 1024)
            {
                bi = 1;
                sm = 512;
            }
            else if (small >= 512)
            {
                bi = 1;
                sm = 256;
            }
            else if (small >= 256)
            {
                bi = 1;
                sm = 128;
            }
            else if (small >= 128)
            {
                bi = 1;
                sm = 64;
            }
            else if (small >= 64)
            {
                bi = 1;
                sm = 32;
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



#if DEBUG
            Trace.TraceInformation("Generating Strategy for {{ rows:{0}, cols:{1} }}", rows, cols);
            Trace.TraceInformation("block {{ x:{0}, y:{1} }} grid {{ x:{2}, y:{3} }}", block.x, block.y, grid.x, grid.y);
#endif
        }




        //        public void GetStrategy(int rows, int cols, out dim3 grid, out dim3 block)
        //        {
        //            if (cols > 1)
        //            {
        //                int x = rows > 1 ? 32 : 1;
        //                int y = 32;
        //                if (cols > 1024)
        //                {
        //                    x = 1;
        //                    y = 1024;
        //                }
        //                else if (cols > 512)
        //                {
        //                    x = 1;
        //                    y = 512;
        //                }
        //                else if (cols > 256)
        //                {
        //                    x = 1;
        //                    y = 256;
        //                }
        //                else if (cols > 128)
        //                {
        //                    x = 1;
        //                    y = 128;
        //                }
        //                else if (cols > 64)
        //                {
        //                    x = 1;
        //                    y = 64;
        //                }
        //                block = new dim3(x, y);
        //                grid = new dim3(Math.Max(1, (int) Math.Floor((double) rows/x)),
        //                    x == 1 ? (int) Math.Max(1, Math.Floor((double) cols/y)) : 1);
        //            }
        //            else
        //            {
        //                int y = cols > 1 ? 32 : 1;
        //                int x = 32;
        //                if (rows > 1024)
        //                {
        //                    y = 1;
        //                    x = 1024;
        //                }
        //                else if (rows > 512)
        //                {
        //                    y = 1;
        //                    x = 512;
        //                }
        //                else if (rows > 256)
        //                {
        //                    y = 1;
        //                    x = 256;
        //                }
        //                else if (rows > 128)
        //                {
        //                    y = 1;
        //                    x = 128;
        //                }
        //                else if (rows > 64)
        //                {
        //                    y = 1;
        //                    x = 64;
        //                }
        //                block = new dim3(x, y);
        //                grid = new dim3(y == 1 ? Math.Max(1, (int) Math.Floor((double) rows/x)) : 1,
        //                    Math.Max(1, (int) Math.Floor((double) cols/y)));
        //            }
        //#if DEBUG
        //            Trace.TraceInformation("Generating Strategy for {{ rows:{0}, cols:{1} }}", rows, cols);
        //            Trace.TraceInformation("block {{ x:{0}, y:{1} }} grid {{ x:{2}, y:{3} }}", block.x, block.y, grid.x, grid.y);
        //#endif
        //        }


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
            return Math.Max(1, (int)Math.Floor(input / 2f));
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