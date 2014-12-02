using System;
using System.Configuration;
using System.IO;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;

namespace CudaRbm.Demo
{
    ///example usage (loads an existing net from bin/net40k, then trains layers higher than 2 using 40000 training records offset 0  records from the beginning): 
    /// -kaggle -net:40k -trainfromlevel:2 -learningrate:0.1 -trainingsize:40000 -skiptrainingrecords:0
    /// get data from https://www.kaggle.com/c/digit-recognizer
    public class Kaggle
    {
        public static float[,] ReadTrainingData(string filePath, int startLine, int count, out int[] labels)
        {
            var ret = new float[count, 784];
            labels = new int[count];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < startLine; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    labels[i] = int.Parse(parts[0]);
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = float.Parse(parts[j + 1]) / 255f;
                    }
                }
            }
            return ret;
        }

        public static float[,] ReadTestData(string filePath, int startLine, int count)
        {
            var ret = new float[count, 784];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < startLine; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = float.Parse(parts[j]) / 255f;
                    }
                }
            }
            return ret;
        }


        public static void Execute()
        {
            Console.WriteLine("Kaggle");
            Console.WriteLine("Getting Host");

            CudafyHost.ClearAllDeviceMemories();
            CudafyHost.ClearDevices();


            GPGPU dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            GPGPUProperties props = dev.GetDeviceProperties();
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            eArchitecture arch = dev.GetArchitecture();
            ePlatform plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            if (plat == ePlatform.x64)
                throw new Exception("CUDA Random will fail currently on x64");

            CudafyModule mod = CudafyTranslator.Cudafy(
                plat,
                arch,
                typeof(ActivationFunctionsCuda),
                typeof(Matrix2DCuda),
                typeof(RestrictedBoltzmannMachineF)
                );

            ThreadOptimiser.Instance = new ThreadOptimiser(props.MultiProcessorCount, props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);


            float learningRate = CommandLine.ReadCommandLine("-learningrate:", float.TryParse, 0.2f);
            int trainingSize = CommandLine.ReadCommandLine("-trainingsize:", int.TryParse, 2048);
            int skipTrainingRecords = CommandLine.ReadCommandLine("-skiptrainingrecords:", int.TryParse, 0);

            GPGPURAND rand = props.Name == "Emulated GPGPU Kernel"
                ? null
                : GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
            DeepBeliefNetworkF dbn = null;
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


                var exitfact = new ManualKeyPressExitEvaluatorFactory<float>(float.Epsilon * 1000, 2000000);

                var net = CommandLine.ReadCommandLine<string>("-net", CommandLine.FakeParseString, null);


                if (net != null)
                {
                    int[] append = CommandLine.ReadCommandLine("-append", CommandLine.ParseIntArray, new int[0]);
                    var d =
                        new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, net));
                    dbn = new DeepBeliefNetworkF(dev, rand, d, learningRate, exitfact, append);
                }
                else
                {
                    dbn = new DeepBeliefNetworkF(
                        dev,
                        rand,
                        new[] { 784, 500, 500, 2000, 10 /*, 234, 156, 104, 70, 47, 32, 22, 15, 10*/},
                        learningRate, exitfact);
                }
                Console.WriteLine("Training Network");
                int[] labels;


                int trainFrom = CommandLine.ReadCommandLine("-trainfromlevel:", int.TryParse, -1);

                if (trainFrom > -1)
                {
                    dbn.TrainLayersFrom(
                        ReadTrainingData(ConfigurationManager.AppSettings["KaggleTrainingData"], skipTrainingRecords,
                            trainingSize,
                            out labels), trainFrom);
                }
                else
                {
                    dbn.TrainAll(
                        ReadTrainingData(ConfigurationManager.AppSettings["KaggleTrainingData"], skipTrainingRecords,
                            trainingSize,
                            out labels));
                }

                if (trainFrom < dbn.NumMachines)
                {
                    var dir = new DirectoryInfo(Environment.CurrentDirectory);

                    DateTime dt = DateTime.Now;

                    DirectoryInfo dir2 =
                        dir.CreateSubdirectory(string.Format("{0:D4}-{1:D2}-{2:D2}_{3:D2}-{4:D2}-{5:D2}", dt.Year,
                            dt.Month,
                            dt.Day, dt.Hour, dt.Minute, dt.Second));
                    int i = 0;
                    foreach (var layerSaveInfo in dbn.GetLayerSaveInfos())
                    {
                        layerSaveInfo.Save(Path.Combine(dir2.FullName, string.Format("layer_{0}.bin", i)));
                        i++;
                    }
                }

                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Reconstructions");
                Console.WriteLine("Training Data:");
                Console.WriteLine();
                //Take a sample of input arrays and try to reconstruct them.
                int[] labels2;
                float[,] tdata = ReadTrainingData(ConfigurationManager.AppSettings["KaggleTrainingData"],
                    skipTrainingRecords + trainingSize,
                    100, out labels2);
                float[,] reconstructedItems =
                    dbn.Reconstruct(tdata);

                reconstructedItems.PrintKaggleMap(labels2, tdata);

                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("Test Data:");
                Console.WriteLine();
                float[,] testData = ReadTestData(ConfigurationManager.AppSettings["KaggleTestData"], 0, 100);
                float[,] reconstructedTestData = dbn.Reconstruct(testData);
                reconstructedTestData.PrintKaggleMap(reference: testData);


                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Daydream");

                do
                {
                    //Day dream 10 images
                    dbn.DayDream(10).PrintKaggleMap();

                    Console.WriteLine();
                    Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                } while (!new[] { 'Q', 'q' }.Contains(Console.ReadKey().KeyChar));
            }
            finally
            {
                dbn.Dispose();
                rand.Dispose();
                dev.UnloadModules();
                dev.FreeAll();
            }
        }
    }


}