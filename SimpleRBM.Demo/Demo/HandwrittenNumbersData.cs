using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using MultidimRBM;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class HandwrittenNumbersData : IDataIO<double>, IDataIO<float>
    {
        public double[,] ReadTrainingData(string filePath, int startLine, int count, out int[] labels)
        {
            string x = File.ReadAllText(filePath);

            x = x.Replace("\r\n", "");

            string[] y = x.Split(" ".ToCharArray());

            double[][] t =
                y.Skip(startLine).Take(count).Select(
                    s =>
                        s.Substring(1).PadRight(1024, '0').Select(
                            n => double.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();
            labels = y.Skip(startLine).Take(count).Select(a => int.Parse(a.Substring(0, 1))).ToArray();
            return Matrix2D.JaggedToMultidimesional(t);
        }

         float[,] IDataIO<float>.ReadTrainingData(string filePath, int startLine, int count, out int[] labels)
        {
            string x = File.ReadAllText(filePath);

            x = x.Replace("\r\n", "");

            string[] y = x.Split(" ".ToCharArray());

            float[][] t =
                y.Skip(startLine).Take(count).Select(
                    s =>
                        s.Substring(1).PadRight(1024, '0').Select(
                            n => float.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();
            labels = y.Skip(startLine).Take(count).Select(a => int.Parse(a.Substring(0, 1))).ToArray();
            return Matrix2D.JaggedToMultidimesional(t);
        }

        public double[,] ReadTestData(string filePath, int startLine, int count)
        {
            string x = File.ReadAllText(filePath);

            x = x.Replace("\r\n", "");

            string[] y = x.Split(" ".ToCharArray());

            double[][] t =
                y.Skip(startLine).Take(count).Select(
                    s =>
                        s.Substring(1).PadRight(1024, '0').Select(
                            n => double.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();
            return Matrix2D.JaggedToMultidimesional(t);
        }

        float[,] IDataIO<float>.ReadTestData(string filePath, int startLine, int count)
        {
            string x = File.ReadAllText(filePath);

            x = x.Replace("\r\n", "");

            string[] y = x.Split(" ".ToCharArray());

            float[][] t =
                y.Skip(startLine).Take(count).Select(
                    s =>
                        s.Substring(1).PadRight(1024, '0').Select(
                            n => float.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();
            return Matrix2D.JaggedToMultidimesional(t);
        }

        public void PrintToScreen(double[,] arr, int[] labels = null, double[,] reference = null, ulong[] keys = null)
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
                    Console.Write(arr[i, j].ToString("N0"));
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


        public void PrintToScreen(float[,] arr, int[] labels = null, float[,] reference = null, ulong[] keys = null)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));


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

        public void PrintMap(float[,] arr)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (j % dataWidth == 0)
                        Console.WriteLine();
                    Console.Write(arr[i, j].ToString("N0"));
                }
                Console.WriteLine();
            }
        }
    }
}