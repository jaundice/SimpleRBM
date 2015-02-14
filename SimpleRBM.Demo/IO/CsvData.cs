using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.IO
{
    public class CsvData : IDataIO<float, string>, IDataIO<double, string>
    {
        public CsvData(string trainingPath, string testPath, bool firstColumnIsLabel, bool firstRowIsHeader)
        {
            FirstColumnIsLabel = firstColumnIsLabel;
            FirstRowIsHeader = firstRowIsHeader;
            TrainingFilePath = trainingPath;
            TestFilePath = testPath;
            if (FirstRowIsHeader)
            {
                using (FileStream fs = File.OpenRead(trainingPath))
                using (var r = new StreamReader(fs))
                {
                    Headers = r.ReadLine().Split(',').ToArray();
                    r.Close();
                    fs.Close();
                }
            }
        }

        public string TestFilePath { get; protected set; }

        public string[] Headers { get; protected set; }

        public bool FirstRowIsHeader { get; protected set; }

        public bool FirstColumnIsLabel { get; protected set; }
        public string TrainingFilePath { get; protected set; }

        double[,] IDataIO<double, string>.ReadTrainingData(int skipRecords, int count, out string[] labels,
            out double[,] labelsCoded)
        {
            labelsCoded = null;
            return Read(TrainingFilePath, skipRecords, count, double.Parse, out labels);
        }

        double[,] IDataIO<double, string>.ReadTestData(int skipRecords, int count)
        {
            string[] labels;
            return Read(TestFilePath, skipRecords, count, double.Parse, out labels);
        }

        void IDataIO<double, string>.PrintToConsole(double[,] arr, double[,] reference, string[] referenceLabels,
            double[,] referenceLabelsCoded, ulong[][] keys, double[,] computedLabels)
        {
            PrintToConsole(arr, reference, referenceLabels, keys);
        }

        void IDataIO<double, string>.PrintToConsole(double[,] arr)
        {
            ((IDataIO<double, string>)this).PrintToConsole(arr, null, null, null, null, null);
        }

        float[,] IDataIO<float, string>.ReadTrainingData(int skipRecords, int count, out string[] labels,
            out float[,] labelsCoded)
        {
            labelsCoded = null;
            return Read(TrainingFilePath, skipRecords, count, float.Parse, out labels);
        }

        float[,] IDataIO<float, string>.ReadTestData(int skipRecords, int count)
        {
            string[] labels;
            return Read(TestFilePath, skipRecords, count, float.Parse, out labels);
        }

        void IDataIO<float, string>.PrintToConsole(float[,] arr, float[,] reference, string[] referenceLabels,
            float[,] referenceLabelsCoded, ulong[][] keys, float[,] computedLabels)
        {
            PrintToConsole(arr, reference, referenceLabels, keys);
        }

        void IDataIO<float, string>.PrintToConsole(float[,] arr)
        {
            ((IDataIO<float, string>)this).PrintToConsole(arr, null, null, null, null, null);
        }

        private T[,] Read<T>(string path, int skipRows, int count, Func<string, T> fieldParser, out string[] ids)
        {
            int cols = FirstColumnIsLabel ? Headers.Length - 1 : Headers.Length;

            List<T[]> rows = new List<T[]>();
            List<string> labels = new List<string>();

            using (FileStream fs = File.OpenRead(path))
            using (var r = new StreamReader(fs))
            {
                if (FirstRowIsHeader)
                    skipRows++;
                for (int i = 0; i < skipRows; i++)
                    r.ReadLine();
                string line;
                int j;
                int readOffset = FirstColumnIsLabel ? 1 : 0;
                for (j = 0, line = r.ReadLine(); j < count && !r.EndOfStream; j++, line = r.ReadLine())
                {
                    string[] parts = line.Split(',').ToArray();
                    if (FirstColumnIsLabel)
                        labels.Add(parts[0]);

                    T[] row = new T[cols];
                    for (int k = 0; k < row.GetLength(0); k++)
                    {
                        row[k] = parts[k + readOffset] == "" ? default(T) : fieldParser(parts[k + readOffset]);
                    }
                    rows.Add(row);
                }
            }

            var temp = rows.ToArray();

            var data = new T[rows.Count, rows[0].Length];
            Parallel.For(0, data.GetLength(0), i =>
            {
                for (var j = 0; j < data.GetLength(1); j++)
                {
                    data[i, j] = temp[i][j];
                }
            });

            ids = labels.ToArray();
            return data;

        }

        private void PrintToConsole<T>(T[,] data, T[,] referenceData, string[] labels, ulong[][] keys)
        {
            for (int row = 0; row < data.GetLength(0); row++)
            {
                if (labels != null)
                    Console.WriteLine(labels[row]);
                if (keys != null)
                {
                    Console.WriteLine(string.Join(" ", keys[row]));
                }

                var rowData = new T[data.GetLength(1)];
                var refData = new T[rowData.Length];
                for (int j = 0; j < rowData.GetLength(0); j++)
                {
                    rowData[j] = data[row, j];

                    if (referenceData != null)
                    {
                        refData[j] = referenceData[row, j];
                    }
                }

                Console.WriteLine(string.Join(",", rowData.Select(a => string.Format("{0:F4}", a))));
                if (referenceData != null)
                {
                    Console.WriteLine(string.Join(",", refData.Select(a => string.Format("{0:F4}", a))));
                }
                Console.WriteLine();
            }
        }
    }
}