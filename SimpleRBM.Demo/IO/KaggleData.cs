using System.IO;
using System.Linq;
using SimpleRBM.Common;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.IO
{
    public class KaggleData : DataIOBase<int>
    {
        private FieldGrayEncoder<int> _labelGrayEncoder;

        public KaggleData(string trainingDataPath, string testDataPath)
            : base(trainingDataPath, testDataPath)
        {
            TestDataPath = testDataPath;
            _labelGrayEncoder = new FieldGrayEncoder<int>(Enumerable.Range(0, 10));
        }

        public string TestDataPath { get; protected set; }


        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out int[] labels,
            out float[,] labelsCoded)
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
                        ret[i, j] = float.Parse(parts[j + 1])/255f;
                    }
                }
            }
            labelsCoded = _labelGrayEncoder.Encode<float>(labels, 1.0f, 0.0f); //FieldGrayEncoder.EncodeLabels<int, float>(labels, 10);
            return ret;
        }

        protected override float[,] ReadTestDataF(string filePath, int skipRecords, int count)
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
                        ret[i, j] = float.Parse(parts[j])/255f;
                    }
                }
            }
            return ret;
        }

        protected override double[,] ReadTestDataD(string filePath, int startLine, int count)
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
                        ret[i, j] = double.Parse(parts[j])/255.0;
                    }
                }
            }
            return ret;
        }

        protected override double[,] ReadTrainingData(string filePath, int startLine, int count, out int[] labels,
            out double[,] labelsCoded)
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
                        ret[i, j] = double.Parse(parts[j + 1])/255.0;
                    }
                }
            }
            labelsCoded = _labelGrayEncoder.Encode<double>(labels, 1.0, 0.0); //FieldGrayEncoder.EncodeLabels<int, double>(labels, 10);
            return ret;
        }
    }
}