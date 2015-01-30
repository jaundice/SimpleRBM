#define USEFLOAT

using System;
using System.Configuration;
using System.Linq;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Common.LearningRate;
using SimpleRBM.Cuda;
using SimpleRBM.Demo.Demo;
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
            new LayerDefinition {VisibleUnits = 1024, HiddenUnits = 500},
            new LayerDefinition {VisibleUnits = 500, HiddenUnits = 500},
            new LayerDefinition {VisibleUnits = 510, HiddenUnits = 2000}
        };

        private static readonly ILayerDefinition[] _defaultKaggleLayerSizes =
        {
            new LayerDefinition {VisibleUnits = 784, HiddenUnits = 500},
            new LayerDefinition {VisibleUnits = 500, HiddenUnits = 500},
            new LayerDefinition {VisibleUnits = 510, HiddenUnits = 2000}
        };

        //private static readonly ILayerDefinition[] _defaultFacesLayerSizes =
        //{
        //    new LayerDefinition {VisibleUnits = 250*250, HiddenUnits = 1800},
        //    new LayerDefinition {VisibleUnits = 1800, HiddenUnits = 6000},
        //    new LayerDefinition {VisibleUnits = 6005, HiddenUnits = 6000}
        //};

        private static readonly ILayerDefinition[] _defaultFacesLayerSizes =
        {
            new LayerDefinition {VisibleUnits = 250*250, HiddenUnits = 1800},
            new LayerDefinition {VisibleUnits = 1800, HiddenUnits = 1800},
            new LayerDefinition {VisibleUnits = 1800, HiddenUnits = 1800}
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
                    new ConstantLearningRate<TElement>(learningRate),
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
                    new ConstantLearningRate<TElement>(learningRate),
                    trainingSize,
                    skipTrainingRecords, true);
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
                    new ConstantLearningRate<TElement>(learningRate),
                    trainingSize,
                    skipTrainingRecords, true);
            }
        }

        private static void Execute<TDataElementType, TLabel>(IDemo demo,
            IDeepBeliefNetworkFactory<TDataElementType> dbnFactory,
            IDataIO<TDataElementType, TLabel> data,
            ILayerDefinition[] defaultLayerSizes,
            ILearningRateCalculator<TDataElementType> learningRate,
            int trainingSize,
            int skipTrainingRecords, bool classify) where TDataElementType : struct, IComparable<TDataElementType>
        {
            demo.Execute(dbnFactory,
                new ManualKeyPressExitEvaluatorFactory<TDataElementType>(0.0005, 5000),
                /*new CompanionDatasetExitConditionEvaluatorFactory<TDataElementType>(null, 10000, 20,
                   new EpochErrorFileTracker<TDataElementType>("main.log"),
                    new EpochErrorFileTracker<TDataElementType>("companion.log")),*/
                /*new ManualKeyPressExitEvaluatorFactory<TDataElementType>(
                    (TDataElementType)Convert.ChangeType(0.0005, typeof(TDataElementType)), 10000),*/
                defaultLayerSizes,
                data,
                learningRate,
                trainingSize,
                skipTrainingRecords, classify);
        }
    }
}