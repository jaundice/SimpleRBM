using System;
using System.IO;
using System.Linq;
using System.Text;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class KaggleData : IDataIO<float>, IDataIO<double>
    {
        double[,] IDataIO<double>.ReadTrainingData(string filePath, int startLine, int count, out int[] labels)
        {
            var ret = new double[count, 784];
            labels = new int[count];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < startLine; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    labels[i] = int.Parse(parts[0]);
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = double.Parse(parts[j + 1])/255d;
                    }
                }
            }
            return ret;
        }

        double[,] IDataIO<double>.ReadTestData(string filePath, int startLine, int count)
        {
            var ret = new double[count, 784];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < startLine; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = double.Parse(parts[j])/255d;
                    }
                }
            }
            return ret;
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
                    Console.Write(arr[i, j].ToString("N0"));
                }
                Console.WriteLine();
            }
        }

        public void PrintToScreen(double[,] arr, int[] labels = null, double[,] reference = null,
            ulong[] keys = null)
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

        float[,] IDataIO<float>.ReadTrainingData(string filePath, int startLine, int count, out int[] labels)
        {
            var ret = new float[count, 784];
            labels = new int[count];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < startLine; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    labels[i] = int.Parse(parts[0]);
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = float.Parse(parts[j + 1])/255f;
                    }
                }
            }
            return ret;
        }

        float[,] IDataIO<float>.ReadTestData(string filePath, int startLine, int count)
        {
            var ret = new float[count, 784];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < startLine; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = float.Parse(parts[j])/255f;
                    }
                }
            }
            return ret;
        }


        public void PrintMap(float[,] arr)
        {
            var dataWidth = (int) Math.Sqrt(arr.GetLength(1));
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (j%dataWidth == 0)
                        Console.WriteLine();
                    Console.Write(arr[i, j].ToString("N0"));
                }
                Console.WriteLine();
            }
        }

        public void PrintToScreen(float[,] arr, int[] labels = null, float[,] reference = null,
            ulong[] keys = null)
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

        private static string GetCharFor(float f)
        {
            return f == 1f ? "1" : f > 0f ? "." : "0";
        }

        private static string GetCharFor(double f)
        {
            return f == 1d ? "1" : f > 0d ? "." : "0";
        }
    }
}