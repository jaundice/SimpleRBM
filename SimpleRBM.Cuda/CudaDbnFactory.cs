using System;
using System.IO;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public class CudaDbnFactory : IDeepBeliefNetworkFactory<float>, IDeepBeliefNetworkFactory<double>
    {
        IDeepBeliefNetwork<double> IDeepBeliefNetworkFactory<double>.Create(DirectoryInfo networkDataDir,
            ILayerDefinition[] appendLayers)
        {
            GPGPU dev;
            GPGPURAND rand;
            CreateDevice(out dev, out rand);

            return new CudaDbnD(dev, rand, networkDataDir, appendLayers);
        }

        IDeepBeliefNetwork<double> IDeepBeliefNetworkFactory<double>.Create(ILayerDefinition[] layerSizes)
        {
            GPGPU dev;
            GPGPURAND rand;
            CreateDevice(out dev, out rand);

            return new CudaDbnD(
                dev,
                rand,
                layerSizes);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(DirectoryInfo networkDataDir,
            ILayerDefinition[] appendLayers)
        {
            GPGPU dev;
            GPGPURAND rand;
            CreateDevice(out dev, out rand);

            return new CudaDbnF(dev, rand, networkDataDir, appendLayers);
        }

        IDeepBeliefNetwork<float> IDeepBeliefNetworkFactory<float>.Create(ILayerDefinition[] layerSizes)
        {
            GPGPU dev;
            GPGPURAND rand;
            CreateDevice(out dev, out rand);

            return new CudaDbnF(
                dev,
                rand,
                layerSizes);
        }

        private void CreateDevice(out GPGPU dev, out GPGPURAND rand)
        {
            CudafyHost.ClearAllDeviceMemories();
            CudafyHost.ClearDevices();


            dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            GPGPUProperties props = dev.GetDeviceProperties();
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            eArchitecture arch = dev.GetArchitecture();
            ePlatform plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            //if (plat == ePlatform.x64)
            //    throw new Exception("CUDA Random will fail currently on x64");

            CudafyModule mod = CudafyTranslator.Cudafy(
                plat,
                arch,
                typeof (ActivationFunctionsCuda),
                typeof (Matrix2DCuda) //,
                //typeof (Matrix2DCudaD)
                //typeof (CudaRbmF),
                //typeof (CudaRbmD)
                );


            ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount,
                props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            rand = props.Name == "Emulated GPGPU Kernel"
                ? null
                : GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

            Console.WriteLine("Loading Module");
            dev.LoadModule(mod);

            Console.WriteLine("Initializing Randoms");
            if (rand != null)
            {
                rand.SetPseudoRandomGeneratorSeed((ulong) DateTime.Now.Ticks);
                rand.GenerateSeeds();
            }
        }
    }
}