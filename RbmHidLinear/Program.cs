using System;
using System.Collections.Generic;
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
                    numTrainingExamples = 5000;
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
                new CudaAdvancedRbmBinary(dev, rand, 0, 178, 150, false),
                new CudaAdvancedRbmBinary(dev, rand, 1, 150, 150, true),
                new CudaAdvancedRbmBinary(dev, rand, 2, 150, 15, true)
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
                    if (b.Epoch%100 == 0)
                    {
                        var recon = ((CudaAdvancedNetwork) a).Reconstruct(tdata, b.Layer);
                        SaveImages(pathBase, string.Format("{0}_{1}_{{0}}_Reconstruction.jpg", b.Layer, b.Epoch), recon);
                    }
                };

                net.LayerTrainComplete += (a, b) =>
                {
                    var m = ((ICudaNetwork<TElement>) a).Machines[b.Layer];
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}.dat", b.Layer, m.NumVisibleNeurons, m.NumHiddenNeurons,
                            typeof (TElement).Name)));
                };

                //batch the data in gpu memory
                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                {
                    double minLR = 1E-04;

                    string[] lbla;
                    TElement[,] codeda;
                    DecayType decayType = DecayType.Power;
                    net.GreedyBatchedTrain(d.ReadTrainingData(0, numTrainingExamples, out lbla, out codeda),
                        10000,
                        new ManualKeyPressExitEvaluatorFactory<TElement>(greedyTracker, (TElement) 0.005, 5000),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, minLR, decayType),
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, minLR, decayType),
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, 1E-3, decayType)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, minLR, decayType),
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, minLR, decayType),
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, 1E-3, decayType)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, minLR, decayType),
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, minLR, decayType),
                            new DecayingLearningRateFactory<TElement>(0.6, 0.99, 1E-3, decayType))
                        );
                }

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
                new CudaAdvancedRbmBinary(dev, rand, 0, 250*250, 500, false),
                new CudaAdvancedRbmBinary(dev, rand, 1, 500, 4000, true),
                new CudaAdvancedRbmBinary(dev, rand, 2, 4000, 4000, true)
            }))
            {
                net.SetDefaultMachineState(SuspendState.Suspended); //keep data in main memory as much as possible

                net.EpochComplete += (a, b) =>
                {
                    if (b.Epoch%100 == 0)
                    {
                        var dreams = ((CudaAdvancedNetwork) a).Daydream(50, b.Layer);
                        SaveImages(pathBase, string.Format("{0}_{1}_{{0}}_Daydream.jpg", b.Layer, b.Epoch), dreams);
                        dreams = null;
                    }
                };

                net.LayerTrainComplete += (a, b) =>
                {
                    var m = ((ICudaNetwork<TElement>) a).Machines[b.Layer];
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}.dat", b.Layer, m.NumVisibleNeurons, m.NumHiddenNeurons,
                            typeof (TElement).Name)));

                    var dreams = ((CudaAdvancedNetwork) a).Daydream(10, b.Layer);
                    SaveImages(pathBase, string.Format("{0}_{1}_{{0}}_TrainEndDaydream.jpg", b.Layer, b.Epoch), dreams);
                    dreams = null;
                };

                IList<string[]> lbl;
                IList<TElement[,]> coded;
                var training = dataProvider.ReadTrainingData(0, numTrainingExamples, 200, out lbl, out coded);
                for (var kk = 0; kk < training.Count; kk++)
                {
                    SaveImages(pathBase, string.Format("TrainingData_{0}_{{0}}.jpg", kk), training[kk]);
                }
                //batch the data into main memory
                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                {
                    TElement minLR = (TElement) 1E-7;
                    DecayType decayType = DecayType.Power;
                    net.GreedyBatchedTrainMem(training,
                        new ManualKeyPressExitEvaluatorFactory<TElement>(greedyTracker, 0.0005f, 50000, 5),
                        new DecayingLearningRateFactory<TElement>(0.003, 0.999, minLR, decayType),
                        new DecayingLearningRateFactory<TElement>(0.003, 0.999, minLR, decayType),
                        new DecayingLearningRateFactory<TElement>(0.003, 0.999, minLR, decayType)
                        );
                }

                var testData = dataProvider.ReadTestData(numTrainingExamples, 500, 50);

                for (var kk = 0; kk < testData.Count; kk++)
                {
                    var reconstructions = net.Reconstruct(testData[kk]);

                    SaveImages(pathBase, string.Format("testData_{0}_{{0}}.jpg", kk), testData[kk]);
                    SaveImages(pathBase, string.Format("reconstructions_{0}_{{0}}.jpg", kk), reconstructions);
                }
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
                    if (b.Epoch%100 == 0)
                    {
                        TElement[,] daydream;
                        if (b.Layer == net.Machines.Count - 1)
                        {
                            TElement[,] labels;
                            daydream = ((CudaAdvancedNetwork) a).DaydreamWithLabels(10, out labels, true, true);

                            dataProvider.PrintToConsole(daydream,
                                computedLabels: labels);
                        }
                        else
                        {
                            daydream = ((CudaAdvancedNetwork) a).Daydream(10, b.Layer);
                        }
                        SaveImages(pathBase, string.Format("{0}_{1}_DayDream_{{0}}.jpg", b.Layer, b.Epoch), daydream);
                    }
                };
                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                    net.GreedyBatchedSupervisedTrain(
                        dataProvider.ReadTrainingData(0, numTrainingExamples, out lbl, out coded),
                        coded, 4000,
                        new ManualKeyPressExitEvaluatorFactory<TElement>(greedyTracker, 0.0005f, 3920),
                        new DecayingLearningRateFactory<TElement>(0.003, 0.9999, 0.000001),
                        new DecayingLearningRateFactory<TElement>(0.003, 0.9999, 0.000001),
                        new DecayingLearningRateFactory<TElement>(0.003, 0.9999, 0.000001));

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
                        b => (byte) (b*255f)));
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


            ePlatform plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            string kernelPath = Path.Combine(Environment.CurrentDirectory,
                string.Format("CudaKernels_{0}.kernel", plat.ToString()));

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
                    typeof (ActivationFunctionsCuda),
                    typeof (Matrix2DCuda)
                    );
                Console.WriteLine("Saving kernels to {0}", kernelPath);
                mod.Serialize(kernelPath);
            }

            ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount,
                props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            rand = GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

            rand.SetPseudoRandomGeneratorSeed((ulong) DateTime.Now.Ticks);
            rand.GenerateSeeds();

            Console.WriteLine("Loading Module");
            dev.LoadModule(mod);
        }
    }
}