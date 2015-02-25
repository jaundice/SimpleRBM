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
    public class FacesData : DataIOBase<string>
    {
        private List<FileInfo> _allTrainingFiles;
        private string[] _allLabelOptions;
        private List<FileInfo> _allTestFiles;

        public FacesData(string trainingDataPath, string testDataPath)
            : base(trainingDataPath, testDataPath)
        {
            var rnd = new Random();
            _allTrainingFiles =
                new DirectoryInfo(trainingDataPath).EnumerateFiles("*.jpg", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a).ToList();

            _allLabelOptions =
                new DirectoryInfo(trainingDataPath).EnumerateDirectories("*", SearchOption.AllDirectories)
                    .Select(a => a.Name)
                    .ToArray();

            _allTestFiles = trainingDataPath == testDataPath
                ? _allTrainingFiles
                : new DirectoryInfo(testDataPath).EnumerateFiles("*.jpg", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a).ToList();
        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out string[] labels,
            out float[,] labelsCoded)
        {
            var files = _allTrainingFiles.Skip(skipRecords)
                .Take(count)
                .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();


            labelsCoded = LabelEncoder.EncodeLabels<string, float>(labels, _allLabelOptions);
            return ImageDataF(files);
        }

        protected override float[,] ReadTestDataF(string filePath, int skipRecords, int count)
        {
            var rnd = new Random();


            List<FileInfo> files = _allTestFiles
                .Skip(skipRecords)
                .Take(count)
                .ToList();

            return ImageDataF(files);
        }


        private static float[,] ImageDataF(IEnumerable<FileInfo> files)
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

        protected override double[,] ReadTestDataD(string filePath, int skipRecords, int count)
        {
            List<FileInfo> files = _allTestFiles
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            return ImageDataD(files);
        }


        private static double[,] ImageDataD(IEnumerable<FileInfo> files)
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

        protected override double[,] ReadTrainingData(string filePath, int skipRecords, int count, out string[] labels,
            out double[,] labelsCoded)
        {
            List<FileInfo> files = _allTrainingFiles
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();

            labelsCoded = LabelEncoder.EncodeLabels<string, double>(labels, _allLabelOptions);

            return ImageDataD(files);
        }
    }
}