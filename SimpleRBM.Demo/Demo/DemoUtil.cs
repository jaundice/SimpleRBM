using System;
using System.IO;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.Demo
{
    public class DemoUtil
    {
        public static void CodeTestData<TDataElement, TLabel>(IDataIO<TDataElement, TLabel> dataProvider, IDeepBeliefNetwork<TDataElement> dbn, string pathBase)
        where TDataElement : struct, IComparable<TDataElement>
        {
            var allData = dataProvider.ReadTestData(0, 185945);

            var encoded = dbn.Encode(allData);
            var kkey = KeyEncoder.CreateBinaryStringKeys(encoded);

            using (var fs = File.OpenWrite(Path.Combine(pathBase, "Encoded.csv")))
            using (var tw = new StreamWriter(fs))
            {
                for (var i = 0; i < allData.GetLength(0); i++)
                {
                    tw.WriteLine(kkey[i]);
                    //tw.WriteLine("{0},\"{1}\"", labess[i], kkey[i]);
                }
            }
        }


        public static void WireEvents<TDataElement, TLabel>(IDataIO<TDataElement, TLabel> dataProvider, bool classify, IDeepBeliefNetwork<TDataElement> dbn,
            string pathBase) where TDataElement : struct, IComparable<TDataElement>
        {
            if (dbn is IDeepBeliefNetworkExtended<TDataElement>)
            {
                var ddd = (IDeepBeliefNetworkExtended<TDataElement>)dbn;
                TDataElement[,] runningtestData = dataProvider.ReadTestData(0, 20);
                SaveImages(pathBase, "OriginalTestData_{0}.jpg", runningtestData);


                ddd.EpochEnd += (sender, args) =>
                {
                    if (args.Epoch % 500 == 0)
                    {
                        TDataElement[,] dream = ddd.DayDream(10, args.Layer);
                        SaveImages(pathBase, string.Format("{0}_{1}_Dream_{{0}}.jpg", args.Layer, args.Epoch), dream);

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
                        SaveImages(pathBase,
                            string.Format("{0}_{1}_Reconstruction_{{0}}.jpg", args.Layer, args.Epoch),
                            reconstructedRunningTestData);
                    }
                };
            }
        }

        public static void DayDream<TDataElement, TLabel>(IDataIO<TDataElement, TLabel> dataProvider,
            IDeepBeliefNetwork<TDataElement> dbn, string pathBase)
            where TDataElement : struct, IComparable<TDataElement>
        {
            TDataElement[,] dreams = dbn.DayDream(10);
            dataProvider.PrintToConsole(dreams);
            SaveImages(pathBase, "Final_Dreams_{0}.jpg", dreams);
        }

        public static void ReconstructTrainingData<TDataElement, TLabel>(IDataIO<TDataElement, TLabel> dataProvider,
            int trainingSize,
            int skipTrainingRecords, bool classify, IDeepBeliefNetwork<TDataElement> dbn, string pathBase)
            where TDataElement : struct, IComparable<TDataElement>
        {
            TLabel[] labels2;
            TDataElement[,] labelsCoded;
            TDataElement[,] tdata = dataProvider.ReadTrainingData(
                skipTrainingRecords + trainingSize,
                100, out labels2, out labelsCoded);


            TDataElement[,] labelsComputed = null;
            TDataElement[,] reconstructedItems = classify
                ? ((IDeepBeliefNetworkExtended<TDataElement>)dbn).ReconstructWithLabels(tdata,
                    out labelsComputed)
                : dbn.Reconstruct(tdata);


            ulong[][] featureKeys = KeyEncoder.GenerateKeys(labelsComputed);

            dataProvider.PrintToConsole(reconstructedItems, tdata, labels2, keys: featureKeys,
                computedLabels: labelsComputed, referenceLabelsCoded: labelsCoded);

            SaveImages(pathBase, "Final_Reconstructions_{0}.jpg", reconstructedItems);
        }

        public static void ReconstructTestData<TDataElement, TLabel>(IDataIO<TDataElement, TLabel> dataProvider,
            bool classify,
            IDeepBeliefNetwork<TDataElement> dbn, string pathBase, out TDataElement[,] labels)
            where TDataElement : struct, IComparable<TDataElement>
        {
            Console.WriteLine("Test Data:");
            Console.WriteLine();
            TDataElement[,] testData = dataProvider.ReadTestData(0, 100);

            TDataElement[,] computedLabels2 = null;

            TDataElement[,] reconstructedTestData = classify
                ? ((IDeepBeliefNetworkExtended<TDataElement>)dbn).ReconstructWithLabels(testData,
                    out computedLabels2)
                : dbn.Reconstruct(testData);
            ulong[][] featKeys2 =
                KeyEncoder.GenerateKeys(computedLabels2);
            dataProvider.PrintToConsole(reconstructedTestData, testData, keys: featKeys2,
                computedLabels: computedLabels2);

            SaveImages(pathBase, "Final_Reconstructions_Test_{0}.jpg", reconstructedTestData);
            labels = computedLabels2;
        }

        public static void DreamWithClass<TDataElement, TLabel>(IDataIO<TDataElement, TLabel> dataProvider,
            TDataElement[,] computedLabels2,
            IDeepBeliefNetwork<TDataElement> dbn) where TDataElement : struct, IComparable<TDataElement>
        {
            Console.WriteLine("Generating 5 examples of each class");

            int numPossibleLabels = computedLabels2.GetLength(1);
            int numExamples = numPossibleLabels * 5;
            int row = 0;
            var labels3 = new TDataElement[numExamples, numPossibleLabels];
            var one = (TDataElement)Convert.ChangeType(1, typeof(TDataElement));
            for (int label = 0; label < numPossibleLabels; label++)
            {
                for (int exNo = 0; exNo < 5; exNo++)
                {
                    labels3[row++, label] = one;
                }
            }

            TDataElement[,] generated =
                ((IDeepBeliefNetworkExtended<TDataElement>)dbn).DaydreamByClass(labels3);

            dataProvider.PrintToConsole(generated, computedLabels: labels3);
        }

        public static void SaveNetwork<TDataElement, TLabel>(string pathBase, IDeepBeliefNetwork<TDataElement> dbn)
            where TDataElement : struct, IComparable<TDataElement>
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

        public static void SaveImages(string pathBase, string nameFormatString, Array data)
        {
            var doubles = data as double[,];
            if (doubles != null)
                SaveImages(pathBase, nameFormatString, doubles);

            var singles = data as float[,];
            if (singles != null)
                SaveImages(pathBase, nameFormatString, singles);
        }

        private static void SaveImages(string pathBase, string nameFormatString, double[,] data)
        {
            Parallel.For(0, data.GetLength(0),
                a =>
                    ImageUtils.SaveImageData(data, a,
                        Path.Combine(pathBase, string.Format(nameFormatString, a)),
                        b => (byte)(b * 255f)));
        }

        private static void SaveImages(string pathBase, string nameFormatString, float[,] data)
        {
            Parallel.For(0, data.GetLength(0),
                a =>
                    ImageUtils.SaveImageData(data, a,
                        Path.Combine(pathBase, string.Format(nameFormatString, a)),
                        b => (byte)(b * 255f)));
        }
    }
}