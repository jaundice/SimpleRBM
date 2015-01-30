using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class HandwrittenNumbersDataD : DataIOBaseD<int>
    {
        private const int ImgDimension = 32;

        public HandwrittenNumbersDataD(string dataPath)
            : base(dataPath)
        {
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
            labelsCoded = LabelEncoder.EncodeLabels<int, double>(labels, 10);
            return data;
        }


        protected override double[,] ReadTestData(string filePath, int skipRecords, int count)
        {
            int[] labels;
            double[,] labelsCoded;
            return ReadTrainingData(filePath, skipRecords, count, out labels, out labelsCoded);
        }
    }
}