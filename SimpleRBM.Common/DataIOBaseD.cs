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

        public void PrintToScreen(double[,] arr, TLabel[] labels = null, double[,] reference = null, ulong[] keys = null)
        {
            var dataWidth = (int) Math.Sqrt(arr.GetLength(1));


            for (int i = 0; i < arr.GetLength(0); i++)
            {
                Console.WriteLine();
                if (labels != null)
                    Console.Write("{0} ", labels[i]);

                if (keys != null)
                    Console.Write(keys[i]);

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

        public virtual double[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels)
        {
            return ReadTrainingData(DataPath, skipRecords, count, out labels);
        }

        public virtual double[,] ReadTestData(int skipRecords, int count)
        {
            return ReadTestData(DataPath, skipRecords, count);
        }

        protected abstract double[,] ReadTrainingData(string filePath, int startLine, int count, out TLabel[] labels);

        protected abstract double[,] ReadTestData(string filePath, int startLine, int count);

        private static string GetCharFor(double f)
        {
            return f > 0.5 ? "1" : f > 0d ? "." : "0";
        }
    }
}