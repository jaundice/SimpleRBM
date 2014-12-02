using System;
using System.Collections.Generic;
using System.Configuration;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;

namespace CudaRbm.Demo
{
    /// <summary>
    /// This class didnt get much attention as my graphics card does not have enough memory to deal with such large data
    /// </summary>
    public class Mnist
    {
        const int ImageCountForTraining = 10;
        private const int ImageCountForTesting = 10;
        private static int _colourComponents;
        private const int ImageDim = 250; //images are 250x250 get data from http://vis-www.cs.umass.edu/lfw/
        private static float[,] ImageData(IEnumerable<FileInfo> files)
        {
            var trainingImageData = ImageUtils.ReadImageData(files);


            float[,] data = null;
            var i = 0;

            foreach (var bytese in trainingImageData)
            {
                if (i == 0)
                {
                    _colourComponents = bytese.Length / (ImageDim * ImageDim);
                    data = new float[ImageCountForTraining, ImageDim * ImageDim];
                }

                for (var j = 0; j < data.GetLength(1); j++)
                {
                    data[i, j] = (1f / 256f) * (float)bytese[j * _colourComponents];
                }
                i++;
            }
            return data;
        }

        public static void Execute()
        {
            var directory = new DirectoryInfo(ConfigurationManager.AppSettings["FacesDirectory"]);
            var files = directory.GetFiles("*.jpg", SearchOption.AllDirectories);
            var trainingFiles = files.Take(ImageCountForTraining);
            var testingFiles = files.Skip(ImageCountForTraining).Take(ImageCountForTesting);

            Console.WriteLine("Getting Host");

            CudafyHost.ClearDevices();


            GPGPU dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            var props = dev.GetDeviceProperties();
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            var arch = dev.GetArchitecture();
            var plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            if (plat == ePlatform.x64)
                throw new Exception("CUDA Random will fail currently on x64");

            CudafyModule mod = CudafyTranslator.Cudafy(
                plat,
                arch,
                typeof(ActivationFunctionsCuda),
                typeof(Matrix2DCuda),
                typeof(RestrictedBoltzmannMachineF)
                );

            ThreadOptimiser.Instance = new ThreadOptimiser(props.MultiProcessorCount, props.MaxThreadsPerBlock, props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            GPGPURAND rand = props.Name == "Emulated GPGPU Kernel" ? (GPGPURAND)null : GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
            try
            {
                Console.WriteLine("Loading Module");
                dev.LoadModule(mod);

                Console.WriteLine("Initializing Randoms");
                if (rand != null)
                {
                    rand.SetPseudoRandomGeneratorSeed((ulong)DateTime.Now.Ticks);
                    rand.GenerateSeeds();
                }


                Console.WriteLine("Building Deep Belief network");

                var trainingData = ImageData(trainingFiles); //need to call this to initialize _colourComponents for now

                var inputLayerSize = ImageDim * ImageDim;


                var rbm = new DeepBeliefNetworkF(
                    dev,
                    rand,
                     new[] { inputLayerSize, inputLayerSize / 8, 256 },
                    0.4f, new EpochCountExitConditionFactory<float>(150));

                Console.WriteLine("Training Network");
                rbm.TrainAll(trainingData);


                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Reconstructions");
                Console.WriteLine();
                //Take a sample of input arrays and try to reconstruct them.
                float[,] reconstructedItems =
                    rbm.Reconstruct(ImageData(testingFiles));


                Console.Write("Cant display data yet");

                //reconstructedItems.PrintMap();


                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Daydream");

                do
                {
                    //Day dream 10 images
                    //rbm.DayDream(10).PrintMap();
                    Console.Write("Cant display data yet");


                    Console.WriteLine();
                    Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                } while (!new[] { 'Q', 'q' }.Contains(Console.ReadKey().KeyChar));
            }
            finally
            {
                rand.Dispose();
                dev.UnloadModules();
                dev.FreeAll();
            }
        }
    }
}
