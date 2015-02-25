using System;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Cuda;

namespace SimpleRBM.Test
{
    [TestClass]
    public abstract class CudaTestBase

    {
        protected static GPGPU _dev;
        protected static GPGPURAND _rand;

        [TestInitialize]
        public void Init()
        {
            _dev.SetCurrentContext();
        }

        static CudaTestBase()
        {
            CudafyHost.ClearAllDeviceMemories();
            CudafyHost.ClearDevices();


            _dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            GPGPUProperties props = _dev.GetDeviceProperties(false);
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            eArchitecture arch = _dev.GetArchitecture();
            ePlatform plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            //if (plat == ePlatform.x64)
            //    throw new Exception("CUDA Random will fail currently on x64");

            CudafyModule mod = CudafyTranslator.Cudafy(
                plat,
                arch,
                typeof(ActivationFunctionsCuda),
                typeof(Matrix2DCuda)
                );


            ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount,
                props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            _rand = GPGPURAND.Create(_dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

            _rand.SetPseudoRandomGeneratorSeed((ulong)DateTime.Now.Ticks);
            _rand.GenerateSeeds();

            Console.WriteLine("Loading Module");
            _dev.LoadModule(mod);
        }

    }
}