using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class MNistData : IDataIO<float, string>
    {
        private static int _colourComponents;

        public float[,] ReadTrainingData(string filePath, int startLine, int count, out string[] labels)
        {
            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories)
                    .Skip(startLine)
                    .Take(count)
                    .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();

            return ImageData(files);
        }

        public float[,] ReadTestData(string filePath, int startLine, int count)
        {
            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories)
                    .Skip(startLine)
                    .Take(count)
                    .ToList();

            return ImageData(files);
        }

        public void PrintToScreen(float[,] arr, string[] labels = null, float[,] reference = null, ulong[] keys = null)
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

        public void PrintMap(float[,] arr)
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

        private static string GetCharFor(float f)
        {
            return f > 0.5f ? "1" : f > 0f ? "." : "0";
        }

        private static string GetCharFor(double f)
        {
            return f > 0.5 ? "1" : f > 0d ? "." : "0";
        }

        private static float[,] ImageData(IEnumerable<FileInfo> files)
        {
            List<FileInfo> lstFiles = files.ToList();

            IEnumerable<float[]> trainingImageData = ImageUtils.ReadImageData(lstFiles, ImageUtils.ConvertRGBToGreyFloat);


            float[,] data = null;
            int i = 0;

            foreach (var bytese in trainingImageData)
            {
                if (i == 0)
                {
                    data = new float[lstFiles.Count, bytese.Length];
                }
                float[] localBytes = bytese;
                Parallel.For(0, bytese.Length, a => { data[i, a] = localBytes[a]; });
                i++;
            }
            return data;
        }
    }
}