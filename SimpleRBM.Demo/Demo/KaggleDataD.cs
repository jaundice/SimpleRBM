using System.IO;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class KaggleDataD : DataIOBaseD<int>
    {
        public KaggleDataD(string trainingDataPath, string testDataPath)
            : base(trainingDataPath)
        {
            TestDataPath = testDataPath;
        }

        public string TestDataPath { get; protected set; }

        public override double[,] ReadTestData(int skipRecords, int count)
        {
            return ReadTestData(TestDataPath, skipRecords, count);
        }

        protected override double[,] ReadTrainingData(string filePath, int startLine, int count, out int[] labels)
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
                        ret[i, j] = double.Parse(parts[j + 1]) / 255.0;
                    }
                }
            }
            return ret;
        }

        protected override double[,] ReadTestData(string filePath, int startLine, int count)
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
                        ret[i, j] = double.Parse(parts[j]) / 255.0;
                    }
                }
            }
            return ret;
        }
    }
}