using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.RAND;
using Cudafy.Translator;

namespace CudaRbm
{
    internal class Program
    {
        private static void Main()
        {
            //Our dataset cosists of images of handwritten digits (0-9)
            //Let's only take 100 of those for training
            float[][] trainingData = DataParser.Parse("optdigits-tra.txt").ToArray();
            Console.WriteLine("Getting Host");

            CudafyHost.ClearDevices();


            GPGPU dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            var props = dev.GetDeviceProperties();
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            var arch = dev.GetArchitecture();
            var plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            if(plat == ePlatform.x64)
                throw new Exception("CUDA Random will fail currently on x64");

            CudafyModule mod = CudafyTranslator.Cudafy(
                plat,
                arch,
                typeof(ActivationFunctions),
                typeof(Matrix2D),
                typeof(RestrictedBoltzmannMachineF)
                );

            GPGPURAND rand = props.Name == "Emulated GPGPU Kernel" ? (GPGPURAND)null : GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
            dev.Synchronize();
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
                dev.Synchronize();


                //Although it is tempting to say that the final hidden layer has 10 features (10 numbers) but let's keep it real.
                Console.WriteLine("Building Deep Belief network");

                var rbm = new DeepBeliefNetworkF(
                    dev,
                    rand,
                    new dim3(8),
                    new dim3(32, 32),
                    new[] { 1024, 512, 256, 128, 64, 10 },
                    //new[] { 1024, 128, 64, 10 },
                    0.1f);

                Console.WriteLine("Training Network");
                rbm.TrainAll(Matrix2D.JaggedToMultidimesional(trainingData.Take(5).ToArray()) /*, 1500, 5*/);


                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Reconstructions");
                Console.WriteLine();
                //Take a sample of input arrays and try to reconstruct them.
                float[,] reconstructedItems =
                    rbm.Reconstruct(Matrix2D.JaggedToMultidimesional(trainingData.Skip(50).Take(200).ToArray()));

                reconstructedItems.PrintMap();


                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Daydream");

                do
                {
                    //Day dream 10 images
                    rbm.DayDream(10).PrintMap();

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
            //}
        }
    }

    public static class DataParser
    {
        public static float[][] Parse(string filePath)
        {
            string x = File.ReadAllText(filePath);

            x = x.Replace("\r\n", "");

            string[] y = x.Split(" ".ToCharArray());

            float[][] t =
                y.Select(
                    s =>
                        s.Substring(1).PadRight(1024, '0').Select(
                            n => float.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();

            return t;
        }
    }

    public static class ExtensionClasses
    {
        public static void PrintMap(this float[,] arr)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (j % dataWidth == 0)
                        Console.WriteLine();
                    Console.Write(arr[i, j].ToString("N0"));
                }
                Console.WriteLine();
            }
        }

        //public static void PrintMap(this double[,] mat, int rows)
        //{
        //    mat.ToArray().Flatten().PrintMap(rows);
        //}

        //public static double[] Flatten(this double[][] arr)
        //{
        //    return arr.SelectMany(x => x).ToArray();
        //}
    }
}