using System;
using System.Configuration;
using System.IO;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Common.LearningRate;
using SimpleRBM.Cuda;
using SimpleRBM.Demo;
using SimpleRBM.Demo.IO;
using SimpleRBM.Demo.Util;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;
using faces = SimpleRBM.Demo.IO.FacesDataD;
using kaggle = SimpleRBM.Demo.IO.KaggleDataD;
#endif

namespace CudaNN
{
    internal class Program
    {
        public static class Demos
        {
            public const string Kaggle = "Kaggle";
            public const string Data = "Data";
            public const string Faces = "Faces";
        }

        private static void Main(string[] args)
        {
            string demo = Demos.Faces;

            int numTrainingExamples;

            GPGPU dev;
            GPGPURAND rand;
            InitCuda(out dev, out rand);


            var pathBase = Path.Combine(Environment.CurrentDirectory, DateTime.Now.ToString("u").Replace(':', '-'));

            Directory.CreateDirectory(pathBase);


            switch (demo)
            {
                case "Faces":
                    {
                        numTrainingExamples = 750;
                        FacesDemo(dev, rand, numTrainingExamples, pathBase);
                        break;
                    }
                case "Data":
                    {
                        numTrainingExamples = 185945;
                        CsvDemo(dev, rand, numTrainingExamples, pathBase);
                        break;
                    }
                case "Kaggle":
                    {
                        numTrainingExamples = 40000;
                        KaggleDemo(dev, rand, numTrainingExamples, pathBase);
                        break;
                    }
            }
        }

        private static void CsvDemo(GPGPU dev, GPGPURAND rand, int numTrainingExamples, string pathBase)
        {
            IDataIO<TElement, string> d = new CsvData(ConfigurationManager.AppSettings["CsvDataTraining"],
                ConfigurationManager.AppSettings["CsvDataTest"], true, true);


            using (var net = new CudaAdvancedNetwork(new CudaAdvancedRbmBase[]
            {
                new CudaAdvancedRbmBinary(dev, rand, 0, 178, 120, false),
                new CudaAdvancedRbmBinary(dev, rand, 1, 120, 150, true),
                new CudaAdvancedRbmBinary(dev, rand, 2, 150, 20, true)
            }))
            {
                net.SetDefaultMachineState(SuspendState.Active);
                string[] lbl;
                TElement[,] coded;

                var tdata = d.ReadTestData(0, 50);
                var di = Directory.CreateDirectory(Path.Combine(pathBase, "Original"));
                SaveImages(di.FullName, "Original_TestData_{0}.jpg", tdata);

                net.EpochComplete += (a, b) =>
                {
                    if (b.Epoch % 100 == 0)
                    {
                        var recon = ((CudaAdvancedNetwork)a).Reconstruct(tdata, b.Layer);
                        SaveImages(pathBase, string.Format("{0}_{1}_{{0}}_Reconstruction.jpg", b.Layer, b.Epoch), recon);
                    }
                };

                //batch the data in gpu memory
                using (var greedyTracker = new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                    net.GreedyBatchedTrain(d.ReadTestData(0, numTrainingExamples),
                        10000,
                        new ManualKeyPressExitEvaluatorFactory<TElement>(greedyTracker, (TElement)0.005, 50000),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.0001, 0.9999, 0.000001),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.0001, 0.9999, 0.000001),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.0001, 0.9999, 0.000001));

                var testData = d.ReadTrainingData(0, 200, out lbl, out coded);

                var reconstructions = net.Reconstruct(testData);

                DisplayResults(pathBase, d, reconstructions, testData, lbl);

                IDataIO<TElement, string> d2 = new CsvData(ConfigurationManager.AppSettings["CsvDataTest"],
                    ConfigurationManager.AppSettings["CsvDataTest"], true, true);

                string[] labels;
                TElement[,] lcoded;
                var allDataToCode = d2.ReadTrainingData(0, 185945, out labels, out lcoded);
                var encoded = net.Encode(allDataToCode);
                var kkey = KeyEncoder.CreateBinaryStringKeys(encoded);

                using (var fs = File.OpenWrite(Path.Combine(pathBase, "Encoded.csv")))
                using (var tw = new StreamWriter(fs))
                {
                    for (var i = 0; i < allDataToCode.GetLength(0); i++)
                    {
                        tw.WriteLine("{0},\"{1}\"", labels[i], kkey[i]);
                    }
                }
            }
        }


        private static void FacesDemo(GPGPU dev, GPGPURAND rand, int numTrainingExamples, string pathBase)
        {
            IDataIO<TElement, string> dataProvider =
                new FacesData(ConfigurationManager.AppSettings["FacesDirectory"],
                    ConfigurationManager.AppSettings["FacesTestDirectory"]);

            using (var net = new CudaAdvancedNetwork(new CudaAdvancedRbmBase[]
            {
                //new CudaAdvancedRbmLinearHidden(dev, rand, 0, 250*250, 800, finalMomentum:(TElement)0.7), 
                new CudaAdvancedRbmBinary(dev, rand, 0, 250*250, 1000, false),
                //new CudaAdvancedRbmBinary(dev, rand, 0, 250*250, 1000, true),
                new CudaAdvancedRbmBinary(dev, rand, 1, 1000, 4000, true),
                new CudaAdvancedRbmBinary(dev, rand, 2, 4000, 4000, true)
            }))
            {
                net.SetDefaultMachineState(SuspendState.Suspended); //keep data in main memory as much as possible
                string[] lbl;
                TElement[,] coded;
                net.EpochComplete += (a, b) =>
                {
                    if (b.Epoch % 100 == 0)
                    {
                        var dreams = ((CudaAdvancedNetwork)a).Daydream(10, b.Layer);
                        SaveImages(pathBase, string.Format("{0}_{1}_{{0}}_Daydream.jpg", b.Layer, b.Epoch), dreams);
                    }
                };

                net.LayerTrainComplete += (a, b) =>
                {
                    var dreams = ((CudaAdvancedNetwork)a).Daydream(200, b.Layer);
                    SaveImages(pathBase, string.Format("{0}_{1}_{{0}}_TrainEndDaydream.jpg", b.Layer, b.Epoch), dreams);
                };

                var training = dataProvider.ReadTrainingData(0, numTrainingExamples, out lbl, out coded);
                SaveImages(pathBase, "TrainingData_{0}.jpg", training);
                //batch the data into main memory
                using (var greedyTracker = new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                    net.GreedyBatchedTrainMem(training, 200,
                        new ManualKeyPressExitEvaluatorFactory<TElement>(greedyTracker, 0.0005f, 50000),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.0001, 0.9999, 0.000001),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.0001, 0.9999, 0.000001),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.0001, 0.9999, 0.000001));

                var testData = dataProvider.ReadTrainingData(numTrainingExamples, 200, out lbl, out coded);

                var reconstructions = net.Reconstruct(testData);

                SaveImages(pathBase, "testData_{0}.jpg", testData);
                SaveImages(pathBase, "reconstructions_{0}.jpg", reconstructions);
            }
        }


        private static void KaggleDemo(GPGPU dev, GPGPURAND rand, int numTrainingExamples, string pathBase)
        {
            IDataIO<TElement, int> dataProvider =
                new KaggleData(ConfigurationManager.AppSettings["KaggleTrainingData"],
                    ConfigurationManager.AppSettings["KaggleTestData"]);

            using (var net = new CudaAdvancedNetwork(new CudaAdvancedRbmBase[]
            {
                new CudaAdvancedRbmBinary(dev, rand, 0, 784, 500, false, encodingNoiseLevel: (TElement) 0.5),
                new CudaAdvancedRbmBinary(dev, rand, 1, 500, 500, true),
                new CudaAdvancedRbmBinary(dev, rand, 2, 510, 2000, true)
                //visible buffer expanded by 10 for labeling
            }))
            {
                //keep data in gpu memory as much as possible
                net.SetDefaultMachineState(SuspendState.Active);


                int[] lbl;
                TElement[,] coded;


                net.EpochComplete += (a, b) =>
                {
                    if (b.Epoch % 100 == 0)
                    {
                        TElement[,] daydream;
                        if (b.Layer == net.Machines.Count - 1)
                        {
                            TElement[,] labels;
                            daydream = ((CudaAdvancedNetwork)a).DaydreamWithLabels(10, out labels, true, true);

                            dataProvider.PrintToConsole(daydream,
                                computedLabels: labels);
                        }
                        else
                        {
                            daydream = ((CudaAdvancedNetwork)a).Daydream(10, b.Layer);
                        }
                        SaveImages(pathBase, string.Format("{0}_{1}_DayDream_{{0}}.jpg", b.Layer, b.Epoch), daydream);
                    }
                };
                using (var greedyTracker = new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                    net.GreedyBatchedSupervisedTrain(
                        dataProvider.ReadTrainingData(0, numTrainingExamples, out lbl, out coded),
                        coded, 4000,
                        new ManualKeyPressExitEvaluatorFactory<TElement>(greedyTracker, 0.0005f, 3920),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.003, 0.9999, 0.000001),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.003, 0.9999, 0.000001),
                        new LinearlyDecayingLearningRateFactory<TElement>(0.003, 0.9999, 0.000001));

                int[] testSrcLabels;
                TElement[,] testSourceCoded;
                var testData = dataProvider.ReadTrainingData(numTrainingExamples, 500, out testSrcLabels,
                    out testSourceCoded);

                TElement[,] computedLabels;
                var reconstructions = net.ReconstructWithLabels(testData, out computedLabels, softmaxLabels: true);
                Console.WriteLine("Reconstructions");
                DisplayResults(pathBase, dataProvider, reconstructions, testData, testSrcLabels, testSourceCoded,
                    computedLabels);
                Console.WriteLine("Daydream by class");
            }
        }

        private static void DisplayResults<TLabel>(string pathBase, IDataIO<TElement, TLabel> dataProvider,
            TElement[,] reconstructions, TElement[,] referenceData, TLabel[] labels,
            TElement[,] referenceCode = null, TElement[,] computedCode = null)
        {
            dataProvider.PrintToConsole(reconstructions, referenceData, labels, referenceCode,
                computedLabels: computedCode);
            SaveImages(pathBase, "testData_{0}.jpg", referenceData);
            SaveImages(pathBase, "reconstructions_{0}.jpg", reconstructions);
        }

        private static void SaveImages(string pathBase, string nameFormatString, TElement[,] data)
        {
            Parallel.For(0, data.GetLength(0),
                a =>
                    ImageUtils.SaveImageData(data, a,
                        Path.Combine(pathBase, string.Format(nameFormatString, a)),
                        b => (byte)(b * 255f)));
        }


        private static void InitCuda(out GPGPU dev, out GPGPURAND rand)
        {
            CudafyHost.ClearAllDeviceMemories();
            CudafyHost.ClearDevices();


            dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            GPGPUProperties props = dev.GetDeviceProperties();
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            eArchitecture arch = dev.GetArchitecture();
            if ((uint)arch > 291U)
                arch = eArchitecture.sm_35;

            ePlatform plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            if (plat == ePlatform.x64)
                throw new Exception("CUDA Random will fail currently on x64");
            string kernelPath = Path.Combine(Environment.CurrentDirectory, "CudaKernels.kernel");

            CudafyModule mod;
            if (File.Exists(kernelPath))
            {
                Console.WriteLine("Loading kernels  from {0}", kernelPath);
                mod = CudafyModule.Deserialize(kernelPath);
            }
            else
            {
                Console.WriteLine("Compiling cuda kernels");
                mod = CudafyTranslator.Cudafy(
                   plat,
                   arch,
                   typeof(ActivationFunctionsCuda),
                   typeof(Matrix2DCuda)
                   );
                Console.WriteLine("Saving kernels to {0}", kernelPath);
                mod.Serialize(kernelPath);
            }

            ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount,
                props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            rand = GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

            rand.SetPseudoRandomGeneratorSeed((ulong)DateTime.Now.Ticks);
            rand.GenerateSeeds();

            Console.WriteLine("Loading Module");
            dev.LoadModule(mod);
        }
    }
}