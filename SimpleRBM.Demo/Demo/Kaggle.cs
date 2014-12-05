﻿using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    /// example usage (loads an existing net from bin/net40k, then trains layers higher than 2 using 40000 training records offset 0  records from the beginning): 
    /// -kaggle -net:40k -trainfromlevel:2 -learningrate:0.1 -trainingsize:40000 -skiptrainingrecords:0
    /// get data from https://www.kaggle.com/c/digit-recognizer
    public class Kaggle :IDemo
    {
        public  void Execute<T>(IDeepBeliefNetworkFactory<T> dbnFactory,
            IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, int[] defaultLayerSizes,
            IDataIO<T> dataProvider, T learningRate, int trainingSize, int skipTrainingRecords)
            where T : struct, IComparable<T>
        {
            Console.WriteLine("Kaggle");


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
                int[] labels;


                int trainFrom = CommandLine.ReadCommandLine("-trainfromlevel:", int.TryParse, -1);

                T[,] trainingData = dataProvider.ReadTrainingData(
                    ConfigurationManager.AppSettings["KaggleTrainingData"],
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
                int[] labels2;
                T[,] tdata = dataProvider.ReadTrainingData(ConfigurationManager.AppSettings["KaggleTrainingData"],
                    skipTrainingRecords + trainingSize,
                    100, out labels2);

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
                T[,] testData = dataProvider.ReadTestData(ConfigurationManager.AppSettings["KaggleTestData"], 0, 100);

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

    public class KeyEncoder
    {
        public static ulong[] GenerateKeys<T>(T[,] encoded) where T : struct, IComparable<T>
        {
            var keys = new ulong[encoded.GetLength(0)];
            int width = encoded.GetLength(1);
            Parallel.For(0, keys.Length, a =>
            {
                ulong v = 0;
                for (int i = 0; i < width; i++)
                {
                    if (Comparer<T>.Default.Compare(encoded[a, i], default(T)) > 0)
                    {
                        v |= ((ulong) 1u << (width - i));
                    }
                }
                keys[a] = v;
            });
            return keys;
        }
    }
}