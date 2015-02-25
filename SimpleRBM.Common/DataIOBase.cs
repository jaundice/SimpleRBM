using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimpleRBM.Common
{
    public abstract class DataIOBase<TLabel> : IDataIO<float, TLabel>, IDataIO<double, TLabel>
    {
        protected DataIOBase(string trainingDataPath, string testDataPath)
        {
            TrainingDataPath = trainingDataPath;
            TestDataPath = testDataPath;
        }

        public string TestDataPath { get; protected set; }

        public string TrainingDataPath { get; protected set; }

        public virtual void PrintToConsole(double[,] arr, double[,] reference = null, TLabel[] referenceLabels = null,
            double[,] referenceLabelsCoded = null, ulong[][] keys = null, double[,] computedLabels = null)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));

            for (int i = 0; i < arr.GetLength(0); i++)
            {
                Console.WriteLine();
                if (referenceLabels != null)
                    Console.WriteLine("Label:\t{0} ", referenceLabels[i]);

                if (keys != null)
                {
                    Console.WriteLine("Key:\t{0}", string.Join(" ", keys[i]));
                }


                if (referenceLabelsCoded != null)
                {
                    Console.WriteLine("Reference Label:\t\t{0}", FormatLabel(referenceLabelsCoded, i));
                }

                if (computedLabels != null)
                {
                    Console.WriteLine("Computed Label:\t\t{0}", FormatLabel(computedLabels, i));
                }

                Console.WriteLine();
                var builder1 = new StringBuilder();
                var builder2 = new StringBuilder();

                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    builder1.Append(GetCharFor(arr[i, j]));
                    if (reference != null)
                        builder2.Append(GetCharFor(reference[i, j]));
                }

                string l1 = builder1.ToString();
                string l2 = builder2.ToString();

                for (int line = 0; line < dataWidth; line++)
                {
                    var line1 = new string(l1.Skip(line * dataWidth).Take(dataWidth).ToArray());
                    string line2 = reference == null
                        ? null
                        : new string(l2.Skip(line * dataWidth).Take(dataWidth).ToArray());

                    if (reference == null)
                    {
                        Console.WriteLine(line1);
                    }
                    else
                    {
                        Console.WriteLine("{0}       {1}", line1, line2);
                    }
                }

                Console.WriteLine();
            }
        }

        public virtual void PrintToConsole(double[,] arr)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (j % dataWidth == 0)
                        Console.WriteLine();
                    Console.Write(GetCharFor(arr[i, j]));
                }
                Console.WriteLine();
            }
        }

        public virtual double[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels,
            out double[,] labelsCoded)
        {
            return ReadTrainingData(TrainingDataPath, skipRecords, count, out labels, out labelsCoded);
        }

        double[,] IDataIO<double, TLabel>.ReadTestData(int skipRecords, int count)
        {
            return ReadTestDataD(TestDataPath, skipRecords, count);
        }

        IList<double[,]> IDataIO<double, TLabel>.ReadTrainingData(int skipRecords, int count, int batchSize, out IList<TLabel[]> labels,
            out IList<double[,]> labelsCoded)
        {
            var res = new List<double[,]>();
            var lbls = new List<TLabel[]>();
            var coded = new List<double[,]>();

            for (var recs = 0; recs < count; recs += batchSize)
            {
                var batchLen = batchSize;
                if (recs + batchLen > count)
                {
                    batchLen = count - recs;
                }

                TLabel[] lab;
                double[,] cod;
                var batch = ReadTrainingData(TestDataPath, skipRecords + recs, batchLen, out lab, out cod);
                res.Add(batch);
                lbls.Add(lab);
                coded.Add(cod);
            }
            labels = lbls;
            labelsCoded = coded;
            return res;
        }

        IList<double[,]> IDataIO<double, TLabel>.ReadTestData(int skipRecords, int count, int batchSize)
        {
            List<double[,]> result = new List<double[,]>();
            for (var recs = 0; recs < count; recs += batchSize)
            {
                var batchLen = batchSize;
                if (recs + batchLen > count)
                {
                    batchLen = count - recs;
                }

                var batch = ReadTestDataD(TestDataPath, skipRecords + recs, batchLen);
                result.Add(batch);
            }
            return result;
        }


        public virtual void PrintToConsole(float[,] arr, float[,] reference = null, TLabel[] referenceLabels = null,
            float[,] referenceLabelsCoded = null, ulong[][] keys = null, float[,] computedLabels = null)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));


            for (int i = 0; i < arr.GetLength(0); i++)
            {
                Console.WriteLine();
                if (referenceLabels != null)
                    Console.WriteLine("Label:\t{0} ", referenceLabels[i]);

                if (keys != null)
                    Console.WriteLine("Key:\t{0}", string.Join(" ", keys[i]));

                if (referenceLabelsCoded != null)
                {
                    Console.WriteLine("Reference Label:\t{0}", FormatLabel(referenceLabelsCoded, i));
                }

                if (computedLabels != null)
                {
                    Console.WriteLine("Computed Label:\t\t{0}", FormatLabel(computedLabels, i));
                }

                Console.WriteLine();
                var builder1 = new StringBuilder();
                var builder2 = new StringBuilder();

                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    builder1.Append(GetCharFor(arr[i, j]));
                    if (reference != null)
                        builder2.Append(GetCharFor(reference[i, j]));
                }

                string l1 = builder1.ToString();
                string l2 = builder2.ToString();

                for (int line = 0; line < dataWidth; line++)
                {
                    var line1 = new string(l1.Skip(line * dataWidth).Take(dataWidth).ToArray());
                    string line2 = reference == null
                        ? null
                        : new string(l2.Skip(line * dataWidth).Take(dataWidth).ToArray());

                    if (reference == null)
                    {
                        Console.WriteLine(line1);
                    }
                    else
                    {
                        Console.WriteLine("{0}       {1}", line1, line2);
                    }
                }

                Console.WriteLine();
            }
        }

        public virtual void PrintToConsole(float[,] arr)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (j % dataWidth == 0)
                        Console.WriteLine();
                    Console.Write(GetCharFor(arr[i, j]));
                }
                Console.WriteLine();
            }
        }

        public virtual float[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels,
            out float[,] labelsCoded)
        {
            return ReadTrainingData(TrainingDataPath, skipRecords, count, out labels, out labelsCoded);
        }

        float[,] IDataIO<float, TLabel>.ReadTestData(int skipRecords, int count)
        {
            return ReadTestDataF(TestDataPath, skipRecords, count);
        }

        IList<float[,]> IDataIO<float, TLabel>.ReadTrainingData(int skipRecords, int count, int batchSize, out IList<TLabel[]> labels,
            out IList<float[,]> labelsCoded)
        {
            var res = new List<float[,]>();
            var lbls = new List<TLabel[]>();
            var coded = new List<float[,]>();

            for (var recs = 0; recs < count; recs += batchSize)
            {
                var batchLen = batchSize;
                if (recs + batchLen > count)
                {
                    batchLen = count - recs;
                }

                TLabel[] lab;
                float[,] cod;
                var batch = ReadTrainingData(TestDataPath, skipRecords + recs, batchLen, out lab, out cod);
                res.Add(batch);
                lbls.Add(lab);
                coded.Add(cod);
            }
            labels = lbls;
            labelsCoded = coded;
            return res;
        }

        IList<float[,]> IDataIO<float, TLabel>.ReadTestData(int skipRecords, int count, int batchSize)
        {
            List<float[,]> result = new List<float[,]>();
            for (var recs = 0; recs < count; recs += batchSize)
            {
                var batchLen = batchSize;
                if (recs + batchLen > count)
                {
                    batchLen = count - recs;
                }

                var batch = ReadTestDataF(TestDataPath, skipRecords + recs, batchLen);
                result.Add(batch);
            }
            return result;
        }

        private static string FormatLabel(float[,] labels, int rowIndex)
        {
            var sb = new StringBuilder();
            float max = 0;

            for (int j = 0; j < labels.GetLength(1); j++)
            {
                max = Math.Max(max, labels[rowIndex, j]);
            }

            for (int q = 0; q < labels.GetLength(1); q++)
            {
                float f = labels[rowIndex, q];

                sb.Append(float.IsNaN(f) ? "N" : f == 0f ? "." : f == max || f > 0.5f ? "+" : "-");
            }
            return sb.ToString();
        }

        protected abstract float[,] ReadTrainingData(string filePath, int skipRecords, int count, out TLabel[] labels,
            out float[,] labelsCoded);

        protected abstract float[,] ReadTestDataF(string filePath, int skipRecords, int count);

        protected static string GetCharFor(float f)
        {
            return f > 0.5f ? "+" : f > 0f ? "-" : ".";
        }

        private static string FormatLabel(double[,] labels, int rowIndex)
        {
            var sb = new StringBuilder();
            double max = 0;

            for (int j = 0; j < labels.GetLength(1); j++)
            {
                max = Math.Max(max, labels[rowIndex, j]);
            }

            for (int q = 0; q < labels.GetLength(1); q++)
            {
                double f = labels[rowIndex, q];

                sb.Append(double.IsNaN(f) ? "N" : f == 0f ? "." : f == max || f > 0.5 ? "+" : "-");
            }
            return sb.ToString();
        }

        protected abstract double[,] ReadTrainingData(string filePath, int startLine, int count, out TLabel[] labels,
            out double[,] labelsCoded);

        protected abstract double[,] ReadTestDataD(string filePath, int startLine, int count);

        private static string GetCharFor(double f)
        {
            return f > 0.5 ? "+" : f > 0d ? "-" : ".";
        }
    }
}