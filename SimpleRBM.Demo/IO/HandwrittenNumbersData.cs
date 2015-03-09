using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.IO
{
    public class HandwrittenNumbersData : DataIOBase<int>
    {
        private FieldGrayEncoder<int> _labelGrayEncoder;
        private const int ImgDimension = 32;

        public HandwrittenNumbersData(string dataPath)
            : base(dataPath, dataPath)
        {
            _labelGrayEncoder = new FieldGrayEncoder<int>(Enumerable.Range(0, 10));
        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out int[] labels,
            out float[,] labelsCoded)
        {
            string x = File.ReadAllText(filePath);

            x = Regex.Replace(x, @"\s", "");

            int recordlength = (ImgDimension*ImgDimension) + 1;
            var lbl = new int[count];
            var data = new float[count, ImgDimension*ImgDimension];

            Parallel.For(0, count, a => Parallel.For(0, ImgDimension*ImgDimension, b =>
            {
                data[a, b] = float.Parse(x[((skipRecords + a)*recordlength) + b].ToString(CultureInfo.InvariantCulture));
                lbl[a] =
                    int.Parse(
                        x[((skipRecords + a)*recordlength) + ImgDimension*ImgDimension].ToString(
                            CultureInfo.InvariantCulture));
            }));

            labels = lbl;
            labelsCoded = _labelGrayEncoder.Encode<float>(labels, 1.0f, 0.0f);  //FieldGrayEncoder.EncodeLabels<int, float>(labels, 10);
            return data;
        }

        protected override float[,] ReadTestDataF(string filePath, int skipRecords, int count)
        {
            int[] labels;
            float[,] labelsCoded;
            return ReadTrainingData(filePath, skipRecords, count, out labels, out labelsCoded);
        }

        protected override double[,] ReadTrainingData(string filePath, int skipRecords, int count, out int[] labels,
            out double[,] labelsCoded)
        {
            string x = File.ReadAllText(filePath);

            x = Regex.Replace(x, @"\s", "");

            int recordlength = (ImgDimension*ImgDimension) + 1;
            var lbl = new int[count];
            var data = new double[count, ImgDimension*ImgDimension];

            Parallel.For(0, count, a => Parallel.For(0, ImgDimension*ImgDimension, b =>
            {
                data[a, b] = double.Parse(x[((skipRecords + a)*recordlength) + b].ToString(CultureInfo.InvariantCulture));
                lbl[a] =
                    int.Parse(
                        x[((skipRecords + a)*recordlength) + ImgDimension*ImgDimension].ToString(
                            CultureInfo.InvariantCulture));
            }));

            labels = lbl;
            labelsCoded = _labelGrayEncoder.Encode<double>(labels, 1.0, 0.0);  //FieldGrayEncoder.EncodeLabels<int, double>(labels, 10);
            return data;
        }


        protected override double[,] ReadTestDataD(string filePath, int skipRecords, int count)
        {
            int[] labels;
            double[,] labelsCoded;
            return ReadTrainingData(filePath, skipRecords, count, out labels, out labelsCoded);
        }
    }
}