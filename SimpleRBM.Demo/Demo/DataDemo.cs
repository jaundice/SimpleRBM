using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Mono.CSharp;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.Demo
{
    public class DataDemo : IDemo
    {
        public void Execute<TDataElement, TLabel>(IDeepBeliefNetworkFactory<TDataElement> dbnFactory,
            ILayerDefinition[] defaultLayerSizes, IDataIO<TDataElement, TLabel> dataProvider,
            ILearningRateCalculatorFactory<TDataElement> preTrainLearningRateCalculatorFactory,
            IExitConditionEvaluatorFactory<TDataElement> preTrainExitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TDataElement> fineTrainLearningRateCalculatorFactory,
            IExitConditionEvaluatorFactory<TDataElement> fineTrainExitConditionEvaluatorFactory, int trainingSize,
            int skipTrainingRecords, bool classify = true)
            where TDataElement : struct, IComparable<TDataElement>
        {
            IDeepBeliefNetwork<TDataElement> dbn = null;

            try
            {
                Console.WriteLine("Building Deep Belief network");
                var net = CommandLine.ReadCommandLine<string>("-net", CommandLine.FakeParseString, null);

                int batchSize = CommandLine.ReadCommandLine("-batchsize", int.TryParse, -1);


                if (net != null)
                {
                    ILayerDefinition[] append = CommandLine.ReadCommandLine("-append",
                        CommandLine.ParseLayerDefinitionArray, new ILayerDefinition[0]);
                    var d = new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, net));
                    dbn = dbnFactory.Create(d, append);
                }
                else
                {
                    dbn = dbnFactory.Create(defaultLayerSizes);
                }


                string pathBase = Path.Combine(Environment.CurrentDirectory, Guid.NewGuid().ToString());
                Directory.CreateDirectory(pathBase);
                Directory.CreateDirectory(Path.Combine(pathBase, "Original"));

                DemoUtil.WireEvents(dataProvider, classify, dbn, pathBase);

                Console.WriteLine("Training Network");
                TLabel[] labels;


                int trainFrom = CommandLine.ReadCommandLine("-trainfromlevel:", int.TryParse, -1);

                TDataElement[,] referenceLabelsCoded;
                TDataElement[,] trainingData = dataProvider.ReadTrainingData(
                    skipTrainingRecords,
                    trainingSize,
                    out labels, out referenceLabelsCoded);

                DemoUtil.SaveImages(pathBase, "OriginalTrainingData_{0}.jpg", trainingData);


                if (trainFrom > -1)
                {
                    throw new NotImplementedException();
                    //if (batchSize == -1)
                    //{
                    //    dbn.GreedyTrainLayersFrom(trainingData, trainFrom, preTrainExitConditionEvaluatorFactory,
                    //        preTrainLearningRateCalculatorFactory);
                    //}
                    //else
                    //{
                    //    dbn.GreedyBatchedTrainLayersFrom(trainingData, trainFrom, batchSize,
                    //        preTrainExitConditionEvaluatorFactory, preTrainLearningRateCalculatorFactory);
                    //}
                }
                else
                {
                    if (batchSize == -1)
                    {
                        //classifier
                        var extended = dbn as IDeepBeliefNetworkExtended<TDataElement>;

                        if (classify && extended != null)
                            extended.GreedySupervisedTrainAll(trainingData,
                                referenceLabelsCoded, preTrainExitConditionEvaluatorFactory,
                                preTrainLearningRateCalculatorFactory);
                        else
                            dbn.GreedyTrainAll(trainingData, preTrainExitConditionEvaluatorFactory,
                                preTrainLearningRateCalculatorFactory);
                    }
                    else
                    {
                        //classifier
                        var extended = dbn as IDeepBeliefNetworkExtended<TDataElement>;
                        if (classify && extended != null)
                            extended.GreedyBatchedSupervisedTrainAll(
                                trainingData,
                                referenceLabelsCoded, batchSize, preTrainExitConditionEvaluatorFactory,
                                preTrainLearningRateCalculatorFactory);
                        else
                            dbn.GreedyBatchedTrainAll(trainingData, batchSize, preTrainExitConditionEvaluatorFactory,
                                preTrainLearningRateCalculatorFactory);
                    }
                }

                if (trainFrom < dbn.NumMachines)
                {
                    DemoUtil.SaveNetwork<TDataElement, TLabel>(pathBase, dbn);
                }


                Console.WriteLine("Fine train");
                if (dbn is IDeepBeliefNetworkExtended<TDataElement>)
                {
                    if (!classify)
                    {
                        ((IDeepBeliefNetworkExtended<TDataElement>)dbn).UpDownTrainAll(trainingData, 2,
                            fineTrainExitConditionEvaluatorFactory, fineTrainLearningRateCalculatorFactory);
                    }
                    else
                    {
                        ((IDeepBeliefNetworkExtended<TDataElement>)dbn).UpDownTrainSupervisedAll(trainingData,
                            referenceLabelsCoded, 2, fineTrainExitConditionEvaluatorFactory,
                            fineTrainLearningRateCalculatorFactory);
                    }
                }

                DemoUtil.SaveNetwork<TDataElement, TLabel>(pathBase, dbn);


                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Reconstructions");
                Console.WriteLine("Training Data:");
                Console.WriteLine();

                //Take a sample of input arrays and try to reconstruct them.
                DemoUtil.ReconstructTrainingData(dataProvider, trainingSize, skipTrainingRecords, classify, dbn, pathBase);

                Console.WriteLine();
                Console.WriteLine();

                TDataElement[,] computedLabels2;
                DemoUtil.ReconstructTestData(dataProvider, classify, dbn, pathBase, out computedLabels2);


                if (classify)
                {
                    DemoUtil.DreamWithClass(dataProvider, computedLabels2, dbn);
                }

                DemoUtil.CodeTestData(dataProvider, dbn, pathBase);


                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Daydream");

                do
                {
                    //Day dream 10 images
                    DemoUtil.DayDream(dataProvider, dbn, pathBase);

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