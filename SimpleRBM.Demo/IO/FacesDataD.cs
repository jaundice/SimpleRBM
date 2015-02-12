using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.IO
{
    public class FacesDataD : DataIOBaseD<string>
    {
        public FacesDataD(string dataPath)
            : base(dataPath)
        {

        }

        protected override double[,] ReadTrainingData(string filePath, int skipRecords, int count, out string[] labels, out double[,] labelsCoded)
        {
            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories)
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();

            var allLabelOptions =
                new DirectoryInfo(filePath).EnumerateDirectories("*", SearchOption.AllDirectories)
                    .Select(a => a.Name)
                    .ToArray();

            labelsCoded = LabelEncoder.EncodeLabels<string, double>(labels, allLabelOptions);

            return ImageData(files);
        }

        protected override double[,] ReadTestData(string filePath, int skipRecords, int count)
        {
            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories)
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            return ImageData(files);
        }


        private static double[,] ImageData(IEnumerable<FileInfo> files)
        {
            List<FileInfo> lstFiles = files.ToList();

            IEnumerable<double[]> trainingImageData = ImageUtils.ReadImageData(lstFiles, ImageUtils.ConvertRGBToGreyD);


            double[,] data = null;
            int i = 0;

            foreach (var bytese in trainingImageData)
            {
                if (i == 0)
                {
                    data = new double[lstFiles.Count, bytese.Length];
                }
                double[] localBytes = bytese;
                Parallel.For(0, bytese.Length, a => { data[i, a] = localBytes[a]; });
                i++;
            }
            return data;
        }
    }
}