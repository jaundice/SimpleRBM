using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class HandwrittenNumbersDataF : DataIOBaseF<int>
    {
        private const int ImgDimension = 32;

        public HandwrittenNumbersDataF(string dataPath) : base(dataPath)
        {
        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out int[] labels)
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

            return data;
        }

        protected override float[,] ReadTestData(string filePath, int skipRecords, int count)
        {
            int[] labels;
            return ReadTrainingData(filePath, skipRecords, count, out labels);
        }
    }
}