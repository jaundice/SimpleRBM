using System;
using System.Linq;
using System.Text;

namespace SimpleRBM.Common
{
    public abstract class DataIOBaseD<TLabel> : IDataIO<double, TLabel>
    {
        protected DataIOBaseD(string dataPath)
        {
            DataPath = dataPath;
        }

        public string DataPath { get; protected set; }

        public void PrintToScreen(double[,] arr, double[,] reference = null, TLabel[] referenceLabels = null,
            double[,] referenceLabelsCoded = null, ulong[][] keys = null, double[,] computedLabels = null)
        {
            var dataWidth = (int) Math.Sqrt(arr.GetLength(1));

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
                    var sb = new StringBuilder();
                    sb.Append("Reference Label:\t");
                    for (int q = 0; q < referenceLabelsCoded.GetLength(1); q++)
                    {
                        sb.Append(referenceLabelsCoded[i, q] == 0.0
                            ? "."
                            : referenceLabelsCoded[i, q] < 0.5f ? "-" : "+");
                    }

                    Console.WriteLine(sb.ToString());
                }
                if (computedLabels != null)
                {
                    var sb = new StringBuilder();
                    sb.Append("Computed Label:\t\t");
                    for (int q = 0; q < computedLabels.GetLength(1); q++)
                    {
                        sb.Append(computedLabels[i, q] == 0.0 ? "." : computedLabels[i, q] < 0.5f ? "-" : "+");
                    }

                    Console.WriteLine(sb.ToString());
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
                    var line1 = new string(l1.Skip(line*dataWidth).Take(dataWidth).ToArray());
                    string line2 = reference == null
                        ? null
                        : new string(l2.Skip(line*dataWidth).Take(dataWidth).ToArray());

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

        public void PrintMap(double[,] arr)
        {
            var dataWidth = (int) Math.Sqrt(arr.GetLength(1));
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (j%dataWidth == 0)
                        Console.WriteLine();
                    Console.Write(GetCharFor(arr[i, j]));
                }
                Console.WriteLine();
            }
        }

        public virtual double[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels,
            out double[,] labelsCoded)
        {
            return ReadTrainingData(DataPath, skipRecords, count, out labels, out labelsCoded);
        }

        public virtual double[,] ReadTestData(int skipRecords, int count)
        {
            return ReadTestData(DataPath, skipRecords, count);
        }

        protected abstract double[,] ReadTrainingData(string filePath, int startLine, int count, out TLabel[] labels,
            out double[,] labelsCoded);

        protected abstract double[,] ReadTestData(string filePath, int startLine, int count);

        private static string GetCharFor(double f)
        {
            return f > 0.5 ? "+" : f > 0d ? "-" : ".";
        }
    }
}