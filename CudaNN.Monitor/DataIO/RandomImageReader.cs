﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Demo;

namespace CudaNN.DeepBelief.DataIO
{
    public class RandomImageReader<T> : ImageReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        public RandomImageReader(string directoryPath, bool useGrayCodesForLabels, int dataWidth,
            IEnumerable<string> allLabels, string[] imageExtensions, int totalRecordCount, ImageUtils.ConvertPixel<T> convertFromImage,
            Func<T, byte> convertToImage)
            : base(
                directoryPath, useGrayCodesForLabels, dataWidth, allLabels, imageExtensions, totalRecordCount, convertFromImage,
                convertToImage)
        {
        }

        public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels, Func<T, T> sourceToTargetConverter)
        {
            IEnumerable<string> files =
                Directory.EnumerateFiles(DirectoryPath,
                    string.Join("|", ValidImageExtensions.Select(a => string.Format((string)"*{0}", (object)a))),
                    SearchOption.AllDirectories);

            labelsEncoded = new T[count, LabelDataWidth];
            labels = new string[count];
            var data = new T[count, DataWidth];
            var on = (T)Convert.ChangeType(1, typeof(T));
            var off = (T)Convert.ChangeType(0, typeof(T));
            Random rnd = new Random(DateTime.Now.Millisecond);
            var i = 0;
            var cutoff = (double)count / TotalRecordCount;
            foreach (var file in files)
            {
                if (rnd.NextDouble() > cutoff)
                    continue;

                if (i == count - 1)
                    break;

                var lblName = new FileInfo(file).Directory.Name;
                labels[i] = lblName;
                if (UseGrayCodesForLabels)
                {
                    GrayLabelEncoder.Encode<T>(lblName, labelsEncoded, i, 0, sourceToTargetConverter(on), sourceToTargetConverter(off));
                }
                else
                {
                    labelsEncoded[i, NonGrayEncoderIndexes[lblName]] = sourceToTargetConverter(on);
                }

                CopyImageDataToTarget(data, i, 0, file);
                i++;
            }

            return data;
        }

        public override T[,] Read(int count)
        {
            IEnumerable<string> files =
                            Directory.EnumerateFiles(DirectoryPath,
                                string.Join("|", ValidImageExtensions.Select(a => string.Format((string)"*{0}", (object)a))),
                                SearchOption.AllDirectories);

            var data = new T[count, DataWidth];
            var on = (T)Convert.ChangeType(1, typeof(T));
            var off = (T)Convert.ChangeType(0, typeof(T));
            Random rnd = new Random(DateTime.Now.Millisecond);
            var i = 0;
            var cutoff = (double)count / TotalRecordCount;
            foreach (var file in files)
            {
                if (rnd.NextDouble() > cutoff)
                    continue;

                if (i == count - 1)
                    break;

                CopyImageDataToTarget(data, i, 0, file);
                i++;
            }

            return data;
        }


        public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded, out IList<string[]> labels)
        {
            var res = new List<T[,]>();
            var lbls = new List<string[]>();
            var coded = new List<T[,]>();
            var on = (T)Convert.ChangeType(1, typeof(T));
            var off = (T)Convert.ChangeType(0, typeof(T));

            var cutOff = (double)count / TotalRecordCount;
            Random rnd = new Random(DateTime.Now.Millisecond);
            List<string> files =
               Directory.EnumerateFiles(DirectoryPath,
                   string.Join("|", ValidImageExtensions.Select(a => string.Format("*{0}", a))),
                   SearchOption.AllDirectories)
                   .Where(a => rnd.NextDouble() < cutOff)
                   .Take(count)
                   .ToList();

            var batches = Partition(files, batchSize);
            foreach (var batch in batches)
            {
                var lb = new string[batch.Count];
                var cod = new T[batch.Count, LabelDataWidth];
                var data = new T[batch.Count, DataWidth];

                for (int i = 0; i < batch.Count; i++)
                {
                    string lblName = new FileInfo(batch[i]).Directory.Name;
                    lb[i] = lblName;
                    if (UseGrayCodesForLabels)
                    {
                        GrayLabelEncoder.Encode(lblName, cod, i, 0, on, off);
                    }
                    else
                    {
                        cod[i, NonGrayEncoderIndexes[lblName]] = on;
                    }
                    CopyImageDataToTarget(data, i, 0, batch[i]);

                }
                res.Add(data);
                lbls.Add(lb);
                coded.Add(cod);
            }
            labels = lbls;
            labelsEncoded = coded;
            return res;
        }

        public override IList<T[,]> Read(int count, int batchSize)
        {
            var res = new List<T[,]>();
            var on = (T)Convert.ChangeType(1, typeof(T));
            var off = (T)Convert.ChangeType(0, typeof(T));

            var cutOff = (double)count / TotalRecordCount;
            Random rnd = new Random(DateTime.Now.Millisecond);
            List<string> files =
               Directory.EnumerateFiles(DirectoryPath,
                   string.Join("|", ValidImageExtensions.Select(a => string.Format("*{0}", a))),
                   SearchOption.AllDirectories)
                   .Where(a => rnd.NextDouble() < cutOff)
                   .Take(count)
                   .ToList();

            var batches = Partition(files, batchSize);
            foreach (var batch in batches)
            {
                var data = new T[batch.Count, DataWidth];

                for (int i = 0; i < batch.Count; i++)
                {
                    CopyImageDataToTarget(data, i, 0, batch[i]);
                }
                res.Add(data);
            }
            return res;
        }
        //todo sort out overloads with/without converters

        public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels)
        {
            return ReadWithLabels(count, out labelsEncoded, out labels, a => a);
        }

        public override T[,] Read(int count, Func<T, T> sourceToTargetConverter)
        {
            return Read(count);
        }

        public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded, out IList<string[]> labels, Func<T, T> sourceToTargetConverter)
        {
            return ReadWithLabels(count, batchSize, out labelsEncoded, out labels);
        }

        public override IList<T[,]> Read(int count, int batchSize, Func<T, T> sourceToTargetConverter)
        {
            return Read(count, batchSize);
        }

    }
}