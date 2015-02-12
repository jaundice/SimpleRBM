using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.IO
{
    public class HandwrittenNumbersDataF : DataIOBaseF<int>
    {
        private const int ImgDimension = 32;

        public HandwrittenNumbersDataF(string dataPath)
            : base(dataPath)
        {
        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out int[] labels, out float[,] labelsCoded)
        {
            string x = File.ReadAllText(filePath);

            x = Regex.Replace(x, @"\s", "");

            int recordlength = (ImgDimension * ImgDimension) + 1;
            var lbl = new int[count];
            var data = new float[count, ImgDimension * ImgDimension];

            Parallel.For(0, count, a => Parallel.For(0, ImgDimension * ImgDimension, b =>
            {
                data[a, b] = float.Parse(x[((skipRecords + a) * recordlength) + b].ToString(CultureInfo.InvariantCulture));
                lbl[a] =
                    int.Parse(
                        x[((skipRecords + a) * recordlength) + ImgDimension * ImgDimension].ToString(
                            CultureInfo.InvariantCulture));
            }));

            labels = lbl;
            labelsCoded = LabelEncoder.EncodeLabels<int, float>(labels,10);
            return data;
        }

        protected override float[,] ReadTestData(string filePath, int skipRecords, int count)
        {
            int[] labels;
            float[,] labelsCoded;
            return ReadTrainingData(filePath, skipRecords, count, out labels, out labelsCoded);
        }
    }
}