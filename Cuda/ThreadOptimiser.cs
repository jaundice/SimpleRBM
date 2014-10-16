using System;
using Cudafy;

namespace CudaRbm
{
    public class ThreadOptimiser
    {
        public static ThreadOptimiser Instance;

        public ThreadOptimiser(int maxProcs, int maxThreadsPerBlock, int maxThreadsPerMultiProcessor, dim3 maxGridSize,
            dim3 maxBlockSize)
        {
            MaxProcessors = maxProcs;
            MaxThreadsPerBlock = maxThreadsPerBlock;
            MaxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor;
            MaxGridSize = maxGridSize;
            MaxBlockSize = maxBlockSize;
        }

        public dim3 MaxGridSize { get; protected set; }

        public int MaxProcessors { get; protected set; }
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

            grid = new dim3(8);
            block = new dim3(4, 256);
            return;


        }

        public class Strategy
        {
            public dim3 Grid { get; private set; }
            public dim3 Block { get; private set; }
        }
    }
}