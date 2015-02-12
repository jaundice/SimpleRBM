using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.IO
{
    public class WavData : DataIOBaseD<string>, IDataIO<float, string>
    {
        private readonly int _maxSamples;

        public WavData(string dataPath, int maxSamples)
            : base(dataPath)
        {
            _maxSamples = maxSamples;
        }

        public float[,] ReadTrainingData(int skipRecords, int count, out string[] labels, out float[,] labelsCoded)
        {
            string[] slabels;
            double[,] slabelsCoded;

            double[,] sdata = ReadTrainingData(skipRecords, count, out slabels, out slabelsCoded);

            labels = slabels;
            labelsCoded = ConvertArrayToFloat(slabelsCoded);
            return ConvertArrayToFloat(sdata);
        }


        public new float[,] ReadTestData(int skipRecords, int count)
        {
            return ConvertArrayToFloat(ReadTestData(base.DataPath, skipRecords, count));
        }

        public void PrintToConsole(float[,] arr, float[,] reference = null, string[] referenceLabels = null,
            float[,] referenceLabelsCoded = null, ulong[][] keys = null, float[,] computedLabels = null)
        {
            Console.WriteLine("Can't print audio to screen");
        }

        public void PrintToConsole(float[,] arr)
        {
            Console.WriteLine("Can't print audio to screen");
        }

        public override void PrintToConsole(double[,] arr)
        {
            Console.WriteLine("Can't print audio to screen");
        }

        public override void PrintToConsole(double[,] arr, double[,] reference = null, string[] referenceLabels = null,
            double[,] referenceLabelsCoded = null, ulong[][] keys = null, double[,] computedLabels = null)
        {
            Console.WriteLine("Can't print audio to screen");
        }

        protected override double[,] ReadTrainingData(string filePath, int startLine, int count, out string[] labels,
            out double[,] labelsCoded)
        {
            var rnd = new Random();

            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.wav", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a)
                    .Skip(startLine)
                    .Take(count)
                    .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();

            string[] allLabelOptions =
                new DirectoryInfo(filePath).EnumerateDirectories("*", SearchOption.AllDirectories)
                    .Select(a => a.Name)
                    .ToArray();

            labelsCoded = LabelEncoder.EncodeLabels<string, double>(labels, allLabelOptions);
            return ReadWavData(files);
        }

        private double[,] ReadWavData(List<FileInfo> files)
        {
            var data = new double[files.Count, 2 * _maxSamples];
            for (int i = 0; i < files.Count; i++)
            {
                int idx = i;
                double[][] sample = WavUtil.WavUtil.Read(files[i].FullName);
                Parallel.For(0, Math.Min(sample[0].Length, _maxSamples), a =>
                {
                    if (sample.Length == 1)
                    {
                        data[idx, 2 * a] = sample[0][a];
                        data[idx, (2 * a) + 1] = sample[0][a];
                    }
                    else
                    {
                        data[idx, 2 * a] = sample[0][a];
                        data[idx, (2 * a) + 1] = sample[1][a] / 2;
                    }
                });
                //pad any empty samples
                Parallel.For(sample[0].Length, _maxSamples, a =>
                {
                    if (sample.Length == 1)
                    {
                        data[idx, 2 * a] = 0;
                        data[idx, (2 * a) + 1] = 0;
                    }
                    else
                    {
                        data[idx, 2 * a] = 0;
                        data[idx, (2 * a) + 1] = 0;
                    }
                });
            }
            return data;
        }

        protected override double[,] ReadTestData(string filePath, int startLine, int count)
        {
            var rnd = new Random();

            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.wav", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a)
                    .Skip(startLine)
                    .Take(count)
                    .ToList();

            return ReadWavData(files);
        }

        private float[,] ConvertArrayToFloat(double[,] src)
        {
            var arr = new float[src.GetLength(0), src.GetLength(1)];

            Parallel.For(0, arr.GetLength(0),
                a => Parallel.For(0, arr.GetLength(1), b => { arr[a, b] = (float)src[a, b]; }));
            return arr;
        }

        private double[,] ConvertArrayToDouble(float[,] src)
        {
            var arr = new double[src.GetLength(0), src.GetLength(1)];

            Parallel.For(0, arr.GetLength(0),
                a => Parallel.For(0, arr.GetLength(1), b => { arr[a, b] = src[a, b]; }));
            return arr;
        }
    }
}