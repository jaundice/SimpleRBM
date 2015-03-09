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
        public enum ConversionMode
        {
            RgbToGreyInt,
            RgbToGreyPosReal,
            RgbToGreyPosNegReal
        }

        private List<FileInfo> _allTrainingFiles;
        private List<FileInfo> _allTestFiles;

        private ImageUtils.ConvertPixel<float> _floatConverter;
        private ImageUtils.ConvertPixel<double> _doubleConverter;
        private FieldGrayEncoder<string> _labelGrayEncoder;


        public FacesData(string trainingDataPath, string testDataPath,
            ConversionMode mode = ConversionMode.RgbToGreyPosReal)
            : base(trainingDataPath, testDataPath)
        {
            var rnd = new Random();
            _allTrainingFiles =
                new DirectoryInfo(trainingDataPath).EnumerateFiles("*.jpg", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a).ToList();

            var allLabelOptions =
                new DirectoryInfo(trainingDataPath).EnumerateDirectories("*", SearchOption.AllDirectories)
                    .Select(a => a.Name)
                    .ToArray();

            _labelGrayEncoder = new FieldGrayEncoder<string>(allLabelOptions);

            _allTestFiles = trainingDataPath == testDataPath
                ? _allTrainingFiles
                : new DirectoryInfo(testDataPath).EnumerateFiles("*.jpg", SearchOption.AllDirectories).Select(a => new
                {
                    a,
                    rnd = rnd.Next()
                }).OrderBy(a => a.rnd).Select(a => a.a).ToList();

            switch (mode)
            {
                case ConversionMode.RgbToGreyPosReal:
                    {
                        _floatConverter = ImageUtils.ConvertRGBToGreyF;
                        _doubleConverter = ImageUtils.ConvertRGBToGreyD;
                        break;
                    }
                case ConversionMode.RgbToGreyInt:
                    {
                        _floatConverter = ImageUtils.ConvertRGBToGreyIntF;
                        _doubleConverter = ImageUtils.ConvertRGBToGreyIntD;
                        break;
                    }
                case ConversionMode.RgbToGreyPosNegReal:
                    {
                        _floatConverter = ImageUtils.ConvertRGBToGreyPosNegF;
                        _doubleConverter = ImageUtils.ConvertRGBToGreyPosNegD;
                        break;
                    }
            }
        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out string[] labels,
            out float[,] labelsCoded)
        {
            var files = _allTrainingFiles.Skip(skipRecords)
                .Take(count)
                .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();


            labelsCoded = _labelGrayEncoder.Encode<float>(labels,1.0f, 0.0f);
            return ImageDataF(files, _floatConverter);
        }

        protected override float[,] ReadTestDataF(string filePath, int skipRecords, int count)
        {
            var rnd = new Random();


            List<FileInfo> files = _allTestFiles
                .Skip(skipRecords)
                .Take(count)
                .ToList();

            return ImageDataF(files, _floatConverter);
        }


        private static float[,] ImageDataF(IEnumerable<FileInfo> files, ImageUtils.ConvertPixel<float> converter)
        {
            List<FileInfo> lstFiles = files.ToList();

            IEnumerable<float[]> trainingImageData = ImageUtils.ReadImageData(lstFiles, converter);


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

            return ImageDataD(files, _doubleConverter);
        }


        private static double[,] ImageDataD(IEnumerable<FileInfo> files, ImageUtils.ConvertPixel<double> converter)
        {
            List<FileInfo> lstFiles = files.ToList();

            IEnumerable<double[]> trainingImageData = ImageUtils.ReadImageData(lstFiles, converter);


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

            labelsCoded = _labelGrayEncoder.Encode<double>(labels, 1.0, 0.0);

            return ImageDataD(files, _doubleConverter);
        }
    }
}