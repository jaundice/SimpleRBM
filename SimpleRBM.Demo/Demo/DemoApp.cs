using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;

namespace SimpleRBM.Demo.Demo
{
    public class DemoApp : IDemo
    {
        public void Execute<TDataElement, TLabel>(IDeepBeliefNetworkFactory<TDataElement> dbnFactory,
            IExitConditionEvaluatorFactory<TDataElement> exitConditionEvaluatorFactory,
            ILayerDefinition[] defaultLayerSizes,
            IDataIO<TDataElement, TLabel> dataProvider, ILearningRateCalculator<TDataElement> learningRateCalculator,
            int trainingSize,
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
                    dbn = dbnFactory.Create(d, append, learningRateCalculator, exitConditionEvaluatorFactory);
                }
                else
                {
                    dbn = dbnFactory.Create(defaultLayerSizes, learningRateCalculator, exitConditionEvaluatorFactory);
                }

                var companionDatasetExitConditionEvaluatorFactory =
                    exitConditionEvaluatorFactory as CompanionDatasetExitConditionEvaluatorFactory<TDataElement>;
                if (companionDatasetExitConditionEvaluatorFactory != null)
                {
                    companionDatasetExitConditionEvaluatorFactory.Dbn = (IDeepBeliefNetworkExtended<TDataElement>) dbn;
                    companionDatasetExitConditionEvaluatorFactory.TestData = dataProvider.ReadTestData(0, 400);
                }
                string pathBase = Path.Combine(Environment.CurrentDirectory, Guid.NewGuid().ToString());
                Directory.CreateDirectory(pathBase);
                Directory.CreateDirectory(Path.Combine(pathBase, "Original"));
                if (dbn is IDeepBeliefNetworkExtended<TDataElement>)
                {
                    var ddd = (IDeepBeliefNetworkExtended<TDataElement>) dbn;
                    TDataElement[,] runningtestData = dataProvider.ReadTestData(0, 20);
                    Task.Run(() =>
                        Parallel.For(0, runningtestData.GetLength(0), kk =>
                            ImageUtils.SaveImageData(runningtestData, kk,
                                Path.Combine(pathBase, "Original",
                                    string.Format("OriginalTestData_{0}.jpg", kk)),
                                a => Convert.ToByte(Convert.ToSingle(a)*255f))
                            ));

                    ddd.EpochEnd += (sender, args) =>
                    {
                        if (args.Epoch%500 == 0)
                        {
                            //Console.WriteLine("daydream:");
                            TDataElement[,] dream = ddd.DayDream(10, args.Layer);
                            Task.Run(() =>
                                Parallel.For(0, dream.GetLength(0), kk =>
                                    ImageUtils.SaveImageData(dream, kk,
                                        Path.Combine(pathBase,
                                            string.Format("{0}_{1}_Dream_{2}.jpg", args.Layer, args.Epoch, kk)),
                                        a => Convert.ToByte(Convert.ToSingle(a)*255f))
                                    ));

                            //dataProvider.PrintToScreen(dream);

                            //Console.WriteLine("recreate");

                            TDataElement[,] reconstructedRunningTestData = null;
                            TDataElement[,] calculatedLabels = null;
                            ulong[][] runningKeys = null;
                            if (args.Layer == dbn.NumMachines - 1)
                            {
                                reconstructedRunningTestData = classify
                                    ? ddd.Classify(runningtestData, out calculatedLabels)
                                    : ddd.Reconstruct(runningtestData);
                                runningKeys = KeyEncoder.GenerateKeys(calculatedLabels);
                            }
                            else
                            {
                                reconstructedRunningTestData = ddd.Reconstruct(runningtestData, args.Layer);
                            }
                            //dataProvider.PrintToScreen(reconstructedRunningTestData, runningtestData, keys: runningKeys,
                            //    computedLabels: calculatedLabels);

                            Task.Run(() =>
                                Parallel.For(0, reconstructedRunningTestData.GetLength(0), kk =>
                                    ImageUtils.SaveImageData(reconstructedRunningTestData, kk,
                                        Path.Combine(pathBase,
                                            string.Format("{0}_{1}_Reconstruction_{2}.jpg", args.Layer, args.Epoch, kk)),
                                        a => Convert.ToByte(Convert.ToSingle(a)*255f))
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
                            ImageUtils.SaveImageData(trainingData, kk,
                                Path.Combine(pathBase, "Original",
                                    string.Format("OriginalTrainingData_{0}.jpg", kk)),
                                a => Convert.ToByte(Convert.ToSingle(a)*255f))
                            ));


                    if (trainFrom > -1)
                    {
                        if (batchSize == -1)
                        {
                            dbn.GreedyTrainLayersFrom(trainingData, trainFrom);
                        }
                        else
                        {
                            dbn.GreedyBatchedTrainLayersFrom(trainingData, trainFrom, batchSize);
                        }
                    }
                    else
                    {
                        if (batchSize == -1)
                        {
                            //classifier
                            if (classify)
                                ((IDeepBeliefNetworkExtended<TDataElement>) dbn).GreedySupervisedTrainAll(trainingData,
                                    referenceLabelsCoded);
                            else
                                dbn.GreedyTrainAll(trainingData);
                        }
                        else
                        {
                            //classifier
                            if (classify)
                                ((IDeepBeliefNetworkExtended<TDataElement>) dbn).GreedyBatchedSupervisedTrainAll(
                                    trainingData,
                                    referenceLabelsCoded, batchSize);
                            else
                                dbn.GreedyBatchedTrainAll(trainingData, batchSize);
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
                        ((IDeepBeliefNetworkExtended<TDataElement>) dbn).UpDownTrainAll(trainingData, 100, 1000,
                            (TDataElement) Convert.ChangeType(0.1, typeof (TDataElement)));
                    }
                    else
                    {
                        ((IDeepBeliefNetworkExtended<TDataElement>) dbn).UpDownTrainSupervisedAll(trainingData,
                            referenceLabelsCoded, 1000, 1000,
                            (TDataElement) Convert.ChangeType(0.1, typeof (TDataElement)));
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

                //float[,] reconstructedItems =
                //    dbn.Reconstruct(tdata);                


                TDataElement[,] labelsComputed = null;
                TDataElement[,] reconstructedItems = classify
                    ? ((IDeepBeliefNetworkExtended<TDataElement>) dbn).Classify(tdata,
                        out labelsComputed)
                    : dbn.Reconstruct(tdata);


                ulong[][] featureKeys = KeyEncoder.GenerateKeys(labelsComputed);

                dataProvider.PrintToScreen(reconstructedItems, tdata, labels2, keys: featureKeys,
                    computedLabels: labelsComputed, referenceLabelsCoded: labelsCoded);

                Task.Run(() =>
                    Parallel.For(0, reconstructedItems.GetLength(0), kk =>
                        ImageUtils.SaveImageData(reconstructedItems, kk,
                            Path.Combine(pathBase,
                                string.Format("Final_Reconstructions_{0}.jpg", kk)),
                            a => Convert.ToByte(Convert.ToSingle(a)*255f))
                        ));

                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("Test Data:");
                Console.WriteLine();
                TDataElement[,] testData = dataProvider.ReadTestData(0, 100);

                TDataElement[,] computedLabels2 = null;

                //c
                TDataElement[,] reconstructedTestData = classify
                    ? ((IDeepBeliefNetworkExtended<TDataElement>) dbn).Classify(testData, out computedLabels2)
                    : dbn.Reconstruct(testData);
                ulong[][] featKeys2 =
                    KeyEncoder.GenerateKeys(computedLabels2);
                ;
                //float[,] reconstructedTestData = dbn.Reconstruct(testData);
                dataProvider.PrintToScreen(reconstructedTestData, testData, keys: featKeys2,
                    computedLabels: computedLabels2);
                Task.Run(() =>
                    Parallel.For(0, reconstructedTestData.GetLength(0), kk =>
                        ImageUtils.SaveImageData(reconstructedTestData, kk,
                            Path.Combine(pathBase,
                                string.Format("Final_Reconstructions_Test_{0}.jpg", kk)),
                            a => Convert.ToByte(Convert.ToSingle(a)*255f))
                        ));

                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
                Console.WriteLine("Daydream");

                do
                {
                    //Day dream 10 images
                    TDataElement[,] dreams = dbn.DayDream(10);
                    dataProvider.PrintToScreen(dreams);
                    Task.Run(() =>
                        Parallel.For(0, dreams.GetLength(0), kk =>
                            ImageUtils.SaveImageData(dreams, kk,
                                Path.Combine(pathBase,
                                    string.Format("Final_Dreams_{0}.jpg", kk)),
                                a => Convert.ToByte(Convert.ToSingle(a)*255f))
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