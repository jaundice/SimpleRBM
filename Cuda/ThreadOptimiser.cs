using System;
using System.Diagnostics;
using Cudafy;

namespace SimpleRBM.Cuda
{
    public class ThreadOptimiser
    {
        public static ThreadOptimiser Instance;

        public ThreadOptimiser(int multiprocessors, int maxThreadsPerBlock, int maxThreadsPerMultiProcessor, dim3 maxGridSize,
            dim3 maxBlockSize)
        {
            MultiProcessorCount = multiprocessors;
            MaxThreadsPerBlock = maxThreadsPerBlock;
            MaxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor;
            MaxGridSize = maxGridSize;
            MaxBlockSize = maxBlockSize;
        }

        public dim3 MaxGridSize { get; protected set; }

        public int MultiProcessorCount { get; protected set; }
        public int MaxThreadsPerBlock { get; protected set; }

        public int MaxThreadsPerMultiProcessor { get; protected set; }
        public dim3 MaxBlockSize { get; protected set; }

        public void GetStrategy(Matrix m, out dim3 grid, out dim3 block)
        {
            GetStrategy(m.GetLength(0), m.GetLength(1), out grid, out block);
        }

        public void GetStrategy(int xReads, int yReads, out dim3 grid, out dim3 block)
        {
            //temp for now
            //todo: work out better heuristic for assigning grids/blocks/threads

            //temp for now
            Trace.TraceInformation("Generating Strategy for {{ xReads:{0}, yReads:{1} }}", xReads, yReads);



            if (yReads < 2)
            {
                grid = new dim3(Math.Max(1, (int)Math.Floor(xReads / 512f)));
                block = new dim3(512, 1);
            }
            else if (xReads < 2)
            {
                grid = new dim3(1, Math.Max(1, (int)Math.Floor(yReads / 512f)));
                block = new dim3(1, 512);
            }
            else
            {
                //grid = new dim3(16);
                //block = new dim3(4, 256);
                if (xReads >= yReads)
                {
                    int small, big;
                    GetDimension(yReads, out small, out big);
                    grid = new dim3(GetGrid((int)Math.Floor(xReads / (float)small)));
                    block = new dim3(small, big);
                }
                else
                {
                    int small, big;
                    GetDimension(xReads, out small, out big);
                    grid = new dim3(1, GetGrid((int)Math.Floor(yReads / (float)small)));
                    block = new dim3(big, small);
                }
            }

            Trace.TraceInformation("block {{ x:{0}, y:{1} }} grid {{ x:{2}, y:{3} }}", block.x, block.y, grid.x, grid.y);
        }

        static int GetGrid(int input)
        {
            return Math.Max(1, (int)Math.Floor(input / 2f));
        }

        static void GetDimension(int width, out int small, out int big)
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