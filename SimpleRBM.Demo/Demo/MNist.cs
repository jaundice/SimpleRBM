using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Cuda;

namespace SimpleRBM.Demo.Demo
{
    /// <summary>
    /// This class didnt get much attention as my graphics card does not have enough memory to deal with such large data
    /// </summary>
    //public class Mnist
    //{
    //    const int ImageCountForTraining = 10;
    //    private const int ImageCountForTesting = 10;
    //    private static int _colourComponents;
    //    private const int ImageDim = 250; //images are 250x250 get data from http://vis-www.cs.umass.edu/lfw/
    //    private static float[,] ImageData(IEnumerable<FileInfo> files, ImageUtils.ConvertPixel<float> converter )
    //    {
    //        var trainingImageData = ImageUtils.ReadImageData(files, converter);


    //        float[,] data = null;
    //        var i = 0;

    //        foreach (var bytese in trainingImageData)
    //        {
    //            if (i == 0)
    //            {
    //                _colourComponents = bytese.Length / (ImageDim * ImageDim);
    //                data = new float[ImageCountForTraining, ImageDim * ImageDim];
    //            }

    //            for (var j = 0; j < data.GetLength(1); j++)
    //            {
    //                data[i, j] = (1f / 256f) * (float)bytese[j * _colourComponents];
    //            }
    //            i++;
    //        }
    //        return data;
    //    }

    //    public unsafe static void Execute()
    //    {
    //        var directory = new DirectoryInfo(ConfigurationManager.AppSettings["FacesDirectory"]);
    //        var files = directory.GetFiles("*.jpg", SearchOption.AllDirectories);
    //        var trainingFiles = files.Take(ImageCountForTraining);
    //        var testingFiles = files.Skip(ImageCountForTraining).Take(ImageCountForTesting);

    //        Console.WriteLine("Getting Host");

    //        CudafyHost.ClearDevices();


    //        GPGPU dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

    //        var props = dev.GetDeviceProperties();
    //        Console.WriteLine(props.Name);

    //        Console.WriteLine("Compiling CUDA module");

    //        var arch = dev.GetArchitecture();
    //        var plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

    //        if (plat == ePlatform.x64)
    //            throw new Exception("CUDA Random will fail currently on x64");

    //        CudafyModule mod = CudafyTranslator.Cudafy(
    //            plat,
    //            arch,
    //            typeof(ActivationFunctionsCuda),
    //            typeof(Matrix2DCuda),
    //            typeof(CudaRbmF)
    //            );

    //        ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount, props.MaxThreadsPerBlock, props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

    //        GPGPURAND rand = props.Name == "Emulated GPGPU Kernel" ? (GPGPURAND)null : GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
    //        try
    //        {
    //            Console.WriteLine("Loading Module");
    //            dev.LoadModule(mod);

    //            Console.WriteLine("Initializing Randoms");
    //            if (rand != null)
    //            {
    //                rand.SetPseudoRandomGeneratorSeed((ulong)DateTime.Now.Ticks);
    //                rand.GenerateSeeds();
    //            }


    //            Console.WriteLine("Building Deep Belief network");

    //            var trainingData = ImageData(trainingFiles, ImageUtils.ConvertRGBToGreyFloat); //need to call this to initialize _colourComponents for now

    //            var inputLayerSize = ImageDim * ImageDim;


    //            var rbm = new CudaDbnF(
    //                dev,
    //                rand,
    //                 new[] { inputLayerSize, inputLayerSize / 8, 256 },
    //                0.4f, new EpochCountExitConditionFactory<float>(150));

    //            Console.WriteLine("Training Network");
    //            rbm.TrainAll(trainingData);


    //            Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
    //            Console.WriteLine("Reconstructions");
    //            Console.WriteLine();
    //            //Take a sample of input arrays and try to reconstruct them.
    //            float[,] reconstructedItems =
    //                rbm.Reconstruct(ImageData(testingFiles, ImageUtils.ConvertRGBToGreyFloat));


    //            Console.Write("Cant display data yet");

    //            //reconstructedItems.PrintMap();


    //            Console.WriteLine();
    //            Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
    //            Console.WriteLine("Daydream");

    //            do
    //            {
    //                //Day dream 10 images
    //                //rbm.DayDream(10).PrintMap();
    //                Console.Write("Cant display data yet");


    //                Console.WriteLine();
    //                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
    //            } while (!new[] { 'Q', 'q' }.Contains(Console.ReadKey().KeyChar));
    //        }
    //        finally
    //        {
    //            rand.Dispose();
    //            dev.UnloadModules();
    //            dev.FreeAll();
    //        }
    //    }
    //}

    public class MNist : IDemo
    {
        public void Execute<T, L>(IDeepBeliefNetworkFactory<T> dbnFactory,
            IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, int[] defaultLayerSizes,
            IDataIO<T, L> dataProvider, T learningRate, int trainingSize, int skipTrainingRecords)
            where T : struct, IComparable<T>
        {
            Console.WriteLine("MNist");


            IDeepBeliefNetwork<T> dbn = null;

            try
            {
                Console.WriteLine("Building Deep Belief network");


                var net = CommandLine.ReadCommandLine<string>("-net", CommandLine.FakeParseString, null);


                if (net != null)
                {
                    int[] append = CommandLine.ReadCommandLine("-append", CommandLine.ParseIntArray, new int[0]);
                    var d = new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, net));
                    dbn = dbnFactory.Create(d, append, learningRate, exitConditionEvaluatorFactory);
                }
                else
                {
                    dbn = dbnFactory.Create(defaultLayerSizes, learningRate, exitConditionEvaluatorFactory);
                }


                Console.WriteLine("Training Network");
                L[] labels;


                int trainFrom = CommandLine.ReadCommandLine("-trainfromlevel:", int.TryParse, -1);

                T[,] trainingData = dataProvider.ReadTrainingData(
                    ConfigurationManager.AppSettings["FacesDirectory"],
                    skipTrainingRecords,
                    trainingSize,
                    out labels);

                if (trainFrom > -1)
                {
                    dbn.TrainLayersFrom(trainingData, trainFrom);
                }
                else
                {
                    dbn.TrainAll(trainingData);
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
                L[] labels2;
                T[,] tdata = dataProvider.ReadTrainingData(ConfigurationManager.AppSettings["FacesDirectory"],
                    skipTrainingRecords + trainingSize,
                    10, out labels2);

                //float[,] reconstructedItems =
                //    dbn.Reconstruct(tdata);                

                T[,] encoded = dbn.Encode(tdata);
                ulong[] featureKeys = KeyEncoder.GenerateKeys(encoded);
                T[,] reconstructedItems = dbn.Decode(encoded);

                dataProvider.PrintToScreen(reconstructedItems, labels2, tdata, featureKeys);

                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("Test Data:");
                Console.WriteLine();
                T[,] testData = dataProvider.ReadTestData(ConfigurationManager.AppSettings["FacesDirectory"], skipTrainingRecords + trainingSize + 10, 10);

                T[,] encoded2 = dbn.Encode(testData);
                ulong[] featKeys2 = KeyEncoder.GenerateKeys(encoded2);
                T[,] reconstructedTestData = dbn.Decode(encoded);
                //float[,] reconstructedTestData = dbn.Reconstruct(testData);
                dataProvider.PrintToScreen(reconstructedTestData, reference: testData, keys: featKeys2);


                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Daydream");

                do
                {
                    //Day dream 10 images
                    dataProvider.PrintToScreen(dbn.DayDream(10));

                    Console.WriteLine();
                    Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                } while (!new[] { 'Q', 'q' }.Contains(Console.ReadKey().KeyChar));
            }
            finally
            {
                var disp = dbn as IDisposable;
                if (disp != null)
                    disp.Dispose();
            }
        }
    }
}
