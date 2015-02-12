#define USEFLOAT

using System;
using System.Configuration;
using System.Linq;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Common.LearningRate;
using SimpleRBM.Cuda;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.IO;
#if USEFLOAT
using TElement = System.Single;

#else
using TElement = System.Double;
#endif

namespace SimpleRBM.Demo
{
    internal class Program
    {
        private static readonly ILayerDefinition[] _defaultHandwrittenLayerSizes =
        {
            new LayerDefinition
            {
                VisibleUnits = 1024,
                HiddenUnits = 500,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 500,
                HiddenUnits = 500,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 510,
                HiddenUnits = 2000,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            }
        };

        private static readonly ILayerDefinition[] _defaultKaggleLayerSizes =
        {
            new LayerDefinition
            {
                VisibleUnits = 784,
                HiddenUnits = 500,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 500,
                HiddenUnits = 500,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 510,
                HiddenUnits = 2000,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            }
        };

        private static readonly ILayerDefinition[] _defaultFacesLayerSizes =
        {
            new LayerDefinition
            {
                VisibleUnits = 250*250,
                HiddenUnits = 1800,
                VisibleActivation = ActivationFunction.SoftPlus,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 1800,
                HiddenUnits = 1800,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 1800,
                HiddenUnits = 1800,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            }
        };

        private static readonly ILayerDefinition[] _defaultAudioLayerSizes =
        {
            new LayerDefinition
            {
                VisibleUnits = 37000,
                HiddenUnits = 1800,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 1800,
                HiddenUnits = 1800,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 1800,
                HiddenUnits = 1800,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            }
        };

        private static readonly ILayerDefinition[] _defaultCsvLayerSizes =
        {
            new LayerDefinition
            {
                VisibleUnits = 178,
                HiddenUnits = 128,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 128,
                HiddenUnits = 128,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            },
            new LayerDefinition
            {
                VisibleUnits = 128,
                HiddenUnits = 32,
                VisibleActivation = ActivationFunction.Sigmoid,
                HiddenActivation = ActivationFunction.Sigmoid
            }
        };

        private static void Main()
        {
            double learningRate = CommandLine.ReadCommandLine("-learningrate:", double.TryParse, 0.2);
            int trainingSize = CommandLine.ReadCommandLine("-trainingsize:", int.TryParse, 2048);
            int skipTrainingRecords = CommandLine.ReadCommandLine("-skiptrainingrecords:", int.TryParse, 0);


            var demo = new DemoApp();
            var factory = new CudaDbnFactory();
            //var factory = new MultiDimDbnFactory();
            Console.WriteLine("Using {0}", factory.GetType().Name);

#if USEFLOAT
            Console.WriteLine("Using single precision");
#else
            Console.WriteLine("Using double precision");
#endif

            if (Environment.GetCommandLineArgs().Contains("-faces"))
            {
                Console.WriteLine("Executing Faces demo");
                Execute<TElement, string>(
                    demo,
                    factory,
                    new IODataTypeProxy<string>(new FacesDataF(ConfigurationManager.AppSettings["FacesDirectory"]),
                        new FacesDataD(ConfigurationManager.AppSettings["FacesDirectory"])),
                    _defaultFacesLayerSizes,
                    new LinearlyDecayingLearningRateFactory<TElement>(learningRate, 0.999999),
                    new LinearlyDecayingLearningRateFactory<TElement>(0.01, 0.99999),
                    trainingSize,
                    skipTrainingRecords, false);
            }
            else if (Environment.GetCommandLineArgs().Contains("-kaggle"))
            {
                Console.WriteLine("Executing Kaggle demo");
                Execute<TElement, int>(
                    demo,
                    factory,
                    new IODataTypeProxy<int>(new KaggleDataF(ConfigurationManager.AppSettings["KaggleTrainingData"],
                        ConfigurationManager.AppSettings["KaggleTestData"]),
                        new KaggleDataD(ConfigurationManager.AppSettings["KaggleTrainingData"],
                            ConfigurationManager.AppSettings["KaggleTestData"])),
                    _defaultKaggleLayerSizes,
                    new LinearlyDecayingLearningRateFactory<TElement>(learningRate, 0.999999),
                    new LinearlyDecayingLearningRateFactory<TElement>(0.01, 0.99999),
                    trainingSize,
                    skipTrainingRecords, true);
            }
            else if (Environment.GetCommandLineArgs().Contains("-data"))
            {
                Console.WriteLine("Executing Data demo");
                Execute<TElement, string>(
                    new DataDemo(),
                    factory,
                    new CsvData(ConfigurationManager.AppSettings["CsvDataTraining"],
                        ConfigurationManager.AppSettings["CsvDataTest"], true, true),
                    _defaultCsvLayerSizes,
                    new LinearlyDecayingLearningRateFactory<TElement>(learningRate, 0.999999),
                    new LinearlyDecayingLearningRateFactory<TElement>(0.01, 0.99999),
                    trainingSize,
                    skipTrainingRecords, false);
            }
            else if (Environment.GetCommandLineArgs().Contains("-audio"))
            {
                Console.WriteLine("Executing Audio demo");
                Execute<TElement, string>(
                    new AudioDemoApp(),
                    factory,
                    new WavData(ConfigurationManager.AppSettings["WavAudioDirectory"], 18500),
                    _defaultAudioLayerSizes,
                    new LinearlyDecayingLearningRateFactory<TElement>(learningRate, 0.999999),
                    new LinearlyDecayingLearningRateFactory<TElement>(0.01, 0.99999),
                    trainingSize,
                    skipTrainingRecords, false);
            }
            else
            {
                Console.WriteLine("Executing Handwritten digits demo");
                Execute<TElement, int>(
                    demo,
                    factory,
                    new IODataTypeProxy<int>(new HandwrittenNumbersDataF("optdigits-tra.txt"),
                        new HandwrittenNumbersDataD("optdigits-tra.txt")),
                    _defaultHandwrittenLayerSizes,
                    new LinearlyDecayingLearningRateFactory<TElement>(learningRate, 0.999999),
                    new LinearlyDecayingLearningRateFactory<TElement>(0.01, 0.99999),
                    trainingSize,
                    skipTrainingRecords, true);
            }
        }

        private static void Execute<TDataElementType, TLabel>(IDemo demo,
            IDeepBeliefNetworkFactory<TDataElementType> dbnFactory,
            IDataIO<TDataElementType, TLabel> data,
            ILayerDefinition[] defaultLayerSizes,
            ILearningRateCalculatorFactory<TDataElementType> preTrainLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TDataElementType> fineTrainLearningRateCalculatorFactory,
            int trainingSize,
            int skipTrainingRecords, bool classify) where TDataElementType : struct, IComparable<TDataElementType>
        {
            demo.Execute(dbnFactory,
                /*new CompanionDatasetExitConditionEvaluatorFactory<TDataElementType>(null, 10000, 20,
                   new EpochErrorFileTracker<TDataElementType>("main.log"),
                    new EpochErrorFileTracker<TDataElementType>("companion.log")),*/
                /*new ManualKeyPressExitEvaluatorFactory<TDataElementType>(
                    (TDataElementType)Convert.ChangeType(0.0005, typeof(TDataElementType)), 10000),*/
                defaultLayerSizes,
                data,
                preTrainLearningRateCalculatorFactory,
                new ManualKeyPressExitEvaluatorFactory<TDataElementType>(0.0005, 20000),
                fineTrainLearningRateCalculatorFactory,
                new ManualKeyPressExitEvaluatorFactory<TDataElementType>(0.0005, 100),
                trainingSize, skipTrainingRecords, classify);
        }
    }
}