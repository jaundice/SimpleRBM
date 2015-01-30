using System.IO;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class KaggleDataF : DataIOBaseF<int>
    {
        public KaggleDataF(string trainingDataPath, string testDataPath)
            : base(trainingDataPath)
        {
            TestDataPath = testDataPath;
        }

        public string TestDataPath { get; protected set; }

        public override float[,] ReadTestData(int skipRecords, int count)
        {
            return ReadTestData(TestDataPath, skipRecords, count);
        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out int[] labels, out float[,] labelsCoded)
        {
            var ret = new float[count, 784];
            labels = new int[count];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < skipRecords; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    labels[i] = int.Parse(parts[0]);
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = float.Parse(parts[j + 1]) / 255f;
                    }
                }
            }
            labelsCoded = LabelEncoder.EncodeLabels<int, float>(labels, 10);
            return ret;
        }

        protected override float[,] ReadTestData(string filePath, int skipRecords, int count)
        {
            var ret = new float[count, 784];
            using (FileStream fs = File.OpenRead(filePath))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); //skip headers
                for (int i = 0; i < skipRecords; i++)
                    sr.ReadLine();

                for (int i = 0; i < count; i++)
                {
                    string line = sr.ReadLine();
                    string[] parts = line.Split(',');
                    for (int j = 0; j < 784; j++)
                    {
                        ret[i, j] = float.Parse(parts[j]) / 255f;
                    }
                }
            }
            return ret;
        }
    }
}