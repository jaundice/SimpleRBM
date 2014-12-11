﻿using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
{
    public class MNistDataF : DataIOBaseF<string>
    {
        public MNistDataF(string dataPath) : base(dataPath)
        {

        }

        protected override float[,] ReadTrainingData(string filePath, int skipRecords, int count, out string[] labels)
        {
            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories)
                    .Skip(skipRecords)
                    .Take(count)
                    .ToList();

            labels = files.Select(a => a.Directory.Name).ToArray();

            return ImageData(files);
        }

        protected override float[,] ReadTestData(string filePath, int skipRecords, int count)
        {
            List<FileInfo> files =
                new DirectoryInfo(filePath).EnumerateFiles("*.jpg", SearchOption.AllDirectories)
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