using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Demo.Demo;
using SimpleRBM.Demo.Util;

namespace SimpleRBM.Demo.IO
{
    public class FacesDataF : DataIOBaseF<string>
    {
        public FacesDataF(string dataPath)
            : base(dataPath)
        {
        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out string[] labels,
            out float[,] labelsCoded)
        {
            var rnd = new Random();

            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a)
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();

            string[] allLabelOptions =
                new DirectoryInfo(filePath).EnumerateDirectories("*", SearchOption.AllDirectories)
                    .Select(a => a.Name)
                    .ToArray();

            labelsCoded = LabelEncoder.EncodeLabels<string, float>(labels, allLabelOptions);
            return ImageData(files);
        }

        protected override float[,] ReadTestData(string filePath, int skipRecords, int count)
        {
            var rnd = new Random();


            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a)
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            return ImageData(files);
        }


        private static float[,] ImageData(IEnumerable<FileInfo> files)
        {
            List<FileInfo> lstFiles = files.ToList();

            IEnumerable<float[]> trainingImageData = ImageUtils.ReadImageData(lstFiles, ImageUtils.ConvertRGBToGreyF);


            float[,] data = null;
            int i = 0;

            foreach (var bytese in trainingImageData)
            {
                if (i == 0)
                {
                    data = new float[lstFiles.Count, bytese.Length];
                }
                float[] localBytes = bytese;
                Parallel.For(0, bytese.Length, a => { data[i, a] = localBytes[a]; });
                i++;
            }
            return data;
        }
    }
}