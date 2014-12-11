using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class MNistDataD : DataIOBaseD<string>
    {
        public MNistDataD(string dataPath)
            : base(dataPath)
        {

        }

        protected override double[,] ReadTrainingData(string filePath, int skipRecords, int count, out string[] labels)
        {
            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories)
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();

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