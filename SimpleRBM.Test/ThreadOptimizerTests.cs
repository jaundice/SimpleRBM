using System;
using System.Collections.Generic;
using Cudafy;
using Cudafy.Host;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Cuda;

namespace CudaTests
{
    [TestClass]
    public class ThreadOptimizerTests
    {
        private readonly List<Tuple<int, int>> gridSizes = new List<Tuple<int, int>>
        {
            new Tuple<int, int>(1, 8000),
            new Tuple<int, int>(8000, 1),
            new Tuple<int, int>(40000, 2000),
            new Tuple<int, int>(2000, 40000),
            new Tuple<int, int>(500, 500),
            new Tuple<int, int>(128, 20),
            new Tuple<int, int>(20, 128),
            new Tuple<int, int>(128, 512),
            new Tuple<int, int>(512, 64),
            new Tuple<int, int>(64, 2000),
            new Tuple<int, int>(2000, 64),
            new Tuple<int, int>(10, 10),
            new Tuple<int, int>(178, 150),
            new Tuple<int, int>(150, 178),
            new Tuple<int, int>(4010, 500),
            new Tuple<int, int>(500, 4010),
        };

        [ClassInitialize]
        public static void InitThreadOptimizer(TestContext context)
        {
            using (GPGPU dev = CudafyHost.GetDevice(eGPUType.Cuda, 0))
            {
                GPGPUProperties props = dev.GetDeviceProperties();
                
                ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount, props.MaxThreadsPerBlock,
                    props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);
            }
        }
        /// <summary>
        /// just used to check the calculated grid and block sizes for sanity (see trace output)
        /// </summary>
        [TestMethod]
        public void TestDifferentGridShapes()
        {
            foreach (var gridSize in gridSizes)
            {
                dim3 grid, block;
                ThreadOptimiser.Instance.GetStrategy(gridSize.Item1, gridSize.Item2, out grid, out block);
            }
        }
    }
}