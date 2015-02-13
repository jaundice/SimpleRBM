using System;
using System.Linq;
using System.Text;

namespace SimpleRBM.Common
{
    public abstract class DataIOBaseF<TLabel> : IDataIO<float, TLabel>
    {
        protected DataIOBaseF(string dataPath)
        {
            DataPath = dataPath;
        }

        public string DataPath { get; protected set; }



        public virtual void PrintToConsole(float[,] arr, float[,] reference = null, TLabel[] referenceLabels = null, float[,] referenceLabelsCoded = null, ulong[][] keys = null, float[,] computedLabels = null)
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

        private static string FormatLabel(float[,] labels, int rowIndex)
        {
            StringBuilder sb = new StringBuilder();
            float max = 0;

            for (var j = 0; j < labels.GetLength(1); j++)
            {
                max = Math.Max(max, labels[rowIndex, j]);
            }

            for (var q = 0; q < labels.GetLength(1); q++)
            {
                float f = labels[rowIndex, q];

                sb.Append(float.IsNaN(f) ? "N" : f == 0f ? "." : f == max || f > 0.5f ? "+" : "-");
            }
            return sb.ToString();
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

        public virtual float[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels, out float[,] labelsCoded)
        {
            return ReadTrainingData(DataPath, skipRecords, count, out labels, out labelsCoded);
        }

        public virtual float[,] ReadTestData(int skipRecords, int count)
        {
            return ReadTestData(DataPath, skipRecords, count);
        }

        protected abstract float[,] ReadTrainingData(string filePath, int skipRecords, int count, out TLabel[] labels, out float[,] labelsCoded);

        protected abstract float[,] ReadTestData(string filePath, int skipRecords, int count);

        protected static string GetCharFor(float f)
        {
            return f > 0.5f ? "+" : f > 0f ? "-" : ".";
        }
    }
}