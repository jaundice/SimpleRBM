using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Demo.Util;
using SimpleRBM.Demo.WavUtil;

namespace SimpleRBM.Demo.Demo
{
    //very experimental
    public class AudioDemoApp : IDemo
    {
        public void Execute<TDataElement, TLabel>(string pathBase, IDeepBeliefNetworkFactory<TDataElement> dbnFactory,
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

                var companionDatasetExitConditionEvaluatorFactory =
                    preTrainExitConditionEvaluatorFactory as CompanionDatasetExitConditionEvaluatorFactory<TDataElement>;
                if (companionDatasetExitConditionEvaluatorFactory != null)
                {
                    companionDatasetExitConditionEvaluatorFactory.Dbn = (IDeepBeliefNetworkExtended<TDataElement>) dbn;
                    companionDatasetExitConditionEvaluatorFactory.TestData = dataProvider.ReadTestData(0, 400);
                }
                Directory.CreateDirectory(pathBase);
                Directory.CreateDirectory(Path.Combine(pathBase, "Original"));
                if (dbn is IDeepBeliefNetworkExtended<TDataElement>)
                {
                    var ddd = (IDeepBeliefNetworkExtended<TDataElement>) dbn;
                    TDataElement[,] runningtestData = dataProvider.ReadTestData(0, 20);
                    Task.Run(() =>
                        Parallel.For(0, runningtestData.GetLength(0), kk =>
                            WavUtil<TDataElement>.SaveWavData(runningtestData, kk,
                                Path.Combine(pathBase, "Original",
                                    string.Format("OriginalTestData_{0}.wav", kk)),
                                a => Convert.ToDouble(a))
                            ));

                    ddd.EpochEnd += (sender, args) =>
                    {
                        if (args.Epoch%500 == 0)
                        {
                            //Console.WriteLine("daydream:");
                            TDataElement[,] dream = ddd.DayDream(10, args.Layer);
                            Task.Run(() =>
                                Parallel.For(0, dream.GetLength(0), kk =>
                                    WavUtil<TDataElement>.SaveWavData(dream, kk,
                                        Path.Combine(pathBase,
                                            string.Format("{0}_{1}_Dream_{2}.wav", args.Layer, args.Epoch, kk)),
                                        a => Convert.ToDouble(a))
                                    ));

                            //dataProvider.PrintToConsole(dream);

                            //Console.WriteLine("recreate");

                            TDataElement[,] reconstructedRunningTestData = null;
                            TDataElement[,] calculatedLabels = null;
                            ulong[][] runningKeys = null;
                            if (args.Layer == dbn.NumMachines - 1)
                            {
                                reconstructedRunningTestData = classify
                                    ? ddd.ReconstructWithLabels(runningtestData, out calculatedLabels)
                                    : ddd.Reconstruct(runningtestData);
                                runningKeys = KeyEncoder.GenerateKeys(calculatedLabels);
                            }
                            else
                            {
                                reconstructedRunningTestData = ddd.Reconstruct(runningtestData, args.Layer);
                            }
                            //dataProvider.PrintToConsole(reconstructedRunningTestData, runningtestData, keys: runningKeys,
                            //    computedLabels: calculatedLabels);

                            Task.Run(() =>
                                Parallel.For(0, reconstructedRunningTestData.GetLength(0), kk =>
                                    WavUtil<TDataElement>.SaveWavData(reconstructedRunningTestData, kk,
                                        Path.Combine(pathBase,
                                            string.Format("{0}_{1}_Reconstruction_{2}.wav", args.Layer, args.Epoch, kk)),
                                        a => Convert.ToDouble(a))
                                    ));
                        }
                    };
                }
                //for (var kk = 0; kk < 1000; kk++)
                //while(false)
                {
                    Console.WriteLine("Training Network");
                    TLabel[] labels;


                    int trainFrom = CommandLine.ReadCommandLine("-trainfromlevel:", int.TryParse, -1);

                    TDataElement[,] referenceLabelsCoded;
                    TDataElement[,] trainingData = dataProvider.ReadTrainingData(
                        skipTrainingRecords,
                        trainingSize,
                        out labels, out referenceLabelsCoded);
                    Task.Run(() =>
                        Parallel.For(0, trainingData.GetLength(0), kk =>
                            WavUtil<TDataElement>.SaveWavData(trainingData, kk,
                                Path.Combine(pathBase, "Original",
                                    string.Format("OriginalTrainingData_{0}.wav", kk)),
                                a => Convert.ToDouble(a))
                            ));


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
                            if (classify)
                                ((IDeepBeliefNetworkExtended<TDataElement>) dbn).GreedySupervisedTrainAll(trainingData,
                                    referenceLabelsCoded, preTrainExitConditionEvaluatorFactory,
                                    preTrainLearningRateCalculatorFactory, CancellationToken.None);
                            else
                                dbn.GreedyTrainAll(trainingData, preTrainExitConditionEvaluatorFactory,
                                    preTrainLearningRateCalculatorFactory, CancellationToken.None);
                        }
                        else
                        {

                            //classifier
                            if (classify)
                                ((IDeepBeliefNetworkExtended<TDataElement>)dbn).GreedyBatchedSupervisedTrainAll(
                                    trainingData,
                                    referenceLabelsCoded, batchSize, preTrainExitConditionEvaluatorFactory,
                                    preTrainLearningRateCalculatorFactory, CancellationToken.None);
                            else
                                dbn.GreedyBatchedTrainAll(trainingData, batchSize, preTrainExitConditionEvaluatorFactory,
                                    preTrainLearningRateCalculatorFactory, CancellationToken.None);
                        }
                    }

                    if (trainFrom < dbn.NumMachines)
                    {
                        var dir = new DirectoryInfo(pathBase);

                        DateTime dt = DateTime.Now;

                        DirectoryInfo dir2 =
                            dir.CreateSubdirectory(string.Format("{0:D4}-{1:D2}-{2:D2}_{3:D2}-{4:D2}-{5:D2}", dt.Year,
                                dt.Month,
                                dt.Day, dt.Hour, dt.Minute, dt.Second));
                        int i = 0;
                        try
                        {
                            foreach (var layerSaveInfo in dbn.GetLayerSaveInfos())
                            {
                                layerSaveInfo.Save(Path.Combine(dir2.FullName, string.Format("layer_{0}.bin", i)));
                                i++;
                            }
                        }
                        catch (OutOfMemoryException)
                        {
                            // big network on x86 :(
                        }
                    }
                    Console.WriteLine("Fine train");
                    if (!classify)
                    {
                        ((IDeepBeliefNetworkExtended<TDataElement>) dbn).UpDownTrainAll(trainingData, 100,
                            fineTrainExitConditionEvaluatorFactory, fineTrainLearningRateCalculatorFactory, CancellationToken.None);
                    }
                    else
                    {
                        ((IDeepBeliefNetworkExtended<TDataElement>) dbn).UpDownTrainSupervisedAll(trainingData,
                            referenceLabelsCoded, 100, fineTrainExitConditionEvaluatorFactory,
                            fineTrainLearningRateCalculatorFactory, CancellationToken.None);
                    }
                }


                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Reconstructions");
                Console.WriteLine("Training Data:");
                Console.WriteLine();
                //Take a sample of input arrays and try to reconstruct them.
                TLabel[] labels2;
                TDataElement[,] labelsCoded;
                TDataElement[,] tdata = dataProvider.ReadTrainingData(
                    skipTrainingRecords + trainingSize,
                    100, out labels2, out labelsCoded);


                TDataElement[,] labelsComputed = null;
                TDataElement[,] reconstructedItems = classify
                    ? ((IDeepBeliefNetworkExtended<TDataElement>) dbn).ReconstructWithLabels(tdata,
                        out labelsComputed)
                    : dbn.Reconstruct(tdata);


                ulong[][] featureKeys = KeyEncoder.GenerateKeys(labelsComputed);

                dataProvider.PrintToConsole(reconstructedItems, tdata, labels2, keys: featureKeys,
                    computedLabels: labelsComputed, referenceLabelsCoded: labelsCoded);

                Task.Run(() =>
                    Parallel.For(0, reconstructedItems.GetLength(0), kk =>
                        WavUtil<TDataElement>.SaveWavData(reconstructedItems, kk,
                            Path.Combine(pathBase,
                                string.Format("Final_Reconstructions_{0}.wav", kk)),
                            a => Convert.ToDouble(a))
                        ));

                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("Test Data:");
                Console.WriteLine();
                TDataElement[,] testData = dataProvider.ReadTestData(0, 100);

                TDataElement[,] computedLabels2 = null;

                TDataElement[,] reconstructedTestData = classify
                    ? ((IDeepBeliefNetworkExtended<TDataElement>) dbn).ReconstructWithLabels(testData, out computedLabels2)
                    : dbn.Reconstruct(testData);
                ulong[][] featKeys2 =
                    KeyEncoder.GenerateKeys(computedLabels2);

                dataProvider.PrintToConsole(reconstructedTestData, testData, keys: featKeys2,
                    computedLabels: computedLabels2);
                Task.Run(() =>
                    Parallel.For(0, reconstructedTestData.GetLength(0), kk =>
                        WavUtil<TDataElement>.SaveWavData(reconstructedTestData, kk,
                            Path.Combine(pathBase,
                                string.Format("Final_Reconstructions_Test_{0}.wav", kk)),
                            a => Convert.ToDouble(a))
                        ));

                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Daydream");

                do
                {
                    //Day dream 10 images
                    TDataElement[,] dreams = dbn.DayDream(10);
                    dataProvider.PrintToConsole(dreams);
                    Task.Run(() =>
                        Parallel.For(0, dreams.GetLength(0), kk =>
                            WavUtil<TDataElement>.SaveWavData(dreams, kk,
                                Path.Combine(pathBase,
                                    string.Format("Final_Dreams_{0}.wav", kk)),
                                a => Convert.ToDouble(a))
                            ));
                    Console.WriteLine();
                    Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                } while (!new[] {'Q', 'q'}.Contains(Console.ReadKey().KeyChar));
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