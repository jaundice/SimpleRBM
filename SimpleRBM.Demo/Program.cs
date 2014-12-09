//#define USEFLOAT

using System;
using System.Configuration;
using System.Linq;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Cuda;
using SimpleRBM.Demo.Demo;
using SimpleRBM.MultiDim;

namespace SimpleRBM.Demo
{
    internal class Program
    {
        private static readonly int[] _defaultHandwrittenLayerSizes = {1024, 500, 500, 2000, 10};
        private static readonly int[] _defaultKaggleLayerSizes = {784, 500, 500, 2000, 64, 10};
        private static readonly int[] _defaultMNistLayerSizes = {250*250, 2000, 2000};

        private static void Main()
        {
            float learningRate = CommandLine.ReadCommandLine("-learningrate:", float.TryParse, 0.2f);
            int trainingSize = CommandLine.ReadCommandLine("-trainingsize:", int.TryParse, 2048);
            int skipTrainingRecords = CommandLine.ReadCommandLine("-skiptrainingrecords:", int.TryParse, 0);


            var demo = new DemoApp();
            //var factory = new CudaDbnFactory();
            var factory = new MultiDimDbnFactory();

            if (Environment.GetCommandLineArgs().Contains("-mnist"))
            {
                Console.WriteLine("Executing MNist demo");
                Execute<
#if USEFLOAT
                    float
#else
                    double
#endif
                    , string>(
                        demo,
                        factory,
                        new IODataTypeProxy<string>(new MNistDataF(ConfigurationManager.AppSettings["FacesDirectory"]),
                            new MNistDataD(ConfigurationManager.AppSettings["FacesDirectory"])),
                        _defaultMNistLayerSizes,
                        learningRate,
                        trainingSize,
                        skipTrainingRecords);
            }
            else if (Environment.GetCommandLineArgs().Contains("-kaggle"))
            {
                Console.WriteLine("Executing Kaggle demo");
                Execute<
#if USEFLOAT
                    float
#else
                    double
#endif
, int>(
                    demo,
                    factory,
                    new IODataTypeProxy<int>(new KaggleDataF(ConfigurationManager.AppSettings["KaggleTrainingData"],
                        ConfigurationManager.AppSettings["KaggleTestData"]),
                        new KaggleDataD(ConfigurationManager.AppSettings["KaggleTrainingData"],
                            ConfigurationManager.AppSettings["KaggleTestData"])),
                    _defaultKaggleLayerSizes,
                    learningRate,
                    trainingSize,
                    skipTrainingRecords);
            }
            else
            {
                Console.WriteLine("Executing Handwritten digits demo");
                Execute<
#if USEFLOAT
                    float
#else
                    double
#endif
, int>(
                    demo,
                    factory,
                    new IODataTypeProxy<int>(new HandwrittenNumbersDataF("optdigits-tra.txt"),
                        new HandwrittenNumbersDataD("optdigits-tra.txt")),
                    _defaultHandwrittenLayerSizes,
                    learningRate,
                    trainingSize,
                    skipTrainingRecords);
            }
        }

        private static void Execute<TDataElementType, TLabel>(IDemo demo,
            IDeepBeliefNetworkFactory<TDataElementType> dbnFactory,
            IDataIO<TDataElementType, TLabel> data,
            int[] defaultLayerSizes,
            double learningRate,
            int trainingSize,
            int skipTrainingRecords) where TDataElementType : struct, IComparable<TDataElementType>
        {
            demo.Execute(dbnFactory,
                new ManualKeyPressExitEvaluatorFactory<TDataElementType>(
                    (TDataElementType) Convert.ChangeType(0.0005, typeof (TDataElementType)), 2000000),
                defaultLayerSizes,
                data,
                (TDataElementType) Convert.ChangeType(learningRate, typeof (TDataElementType)),
                trainingSize,
                skipTrainingRecords);
        }
    }
}