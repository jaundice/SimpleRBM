﻿using System;
using System.IO;
using System.Linq;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class DemoApp : IDemo
    {
        public void Execute<TDataElement, TLabel>(IDeepBeliefNetworkFactory<TDataElement> dbnFactory,
            IExitConditionEvaluatorFactory<TDataElement> exitConditionEvaluatorFactory, int[] defaultLayerSizes,
            IDataIO<TDataElement, TLabel> dataProvider, TDataElement learningRate, int trainingSize,
            int skipTrainingRecords)
            where TDataElement : struct, IComparable<TDataElement>
        {
            IDeepBeliefNetwork<TDataElement> dbn = null;

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
                TLabel[] labels;


                int trainFrom = CommandLine.ReadCommandLine("-trainfromlevel:", int.TryParse, -1);

                TDataElement[,] trainingData = dataProvider.ReadTrainingData(
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
                TLabel[] labels2;
                TDataElement[,] tdata = dataProvider.ReadTrainingData(
                    skipTrainingRecords + trainingSize,
                    100, out labels2);

                //float[,] reconstructedItems =
                //    dbn.Reconstruct(tdata);                

                TDataElement[,] encoded = dbn.Encode(tdata);
                ulong[] featureKeys = KeyEncoder.GenerateKeys(encoded);
                TDataElement[,] reconstructedItems = dbn.Decode(encoded);

                dataProvider.PrintToScreen(reconstructedItems, labels2, tdata, featureKeys);

                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("Test Data:");
                Console.WriteLine();
                TDataElement[,] testData = dataProvider.ReadTestData(0, 100);

                TDataElement[,] encoded2 = dbn.Encode(testData);
                ulong[] featKeys2 = KeyEncoder.GenerateKeys(encoded2);
                TDataElement[,] reconstructedTestData = dbn.Decode(encoded);
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