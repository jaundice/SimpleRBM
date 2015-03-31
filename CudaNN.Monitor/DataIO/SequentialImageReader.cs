using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SimpleRBM.Demo;

namespace CudaNN.DeepBelief.DataIO
{
    public class SequentialImageReader<T> : ImageReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        public SequentialImageReader(string directoryPath, bool useGrayCodesForLabels, int dataWidth,
            IEnumerable<string> allLabels, string[] imageExtensions, int skipCount, int totalRecordCount,
            Func<T, T> sourceToTargetConverter, Func<T, T> targetToSourceConverter, ImageUtils.ConvertPixel<T> pixelConverter )
            : base(
                directoryPath, useGrayCodesForLabels, dataWidth, allLabels, imageExtensions, totalRecordCount,
                sourceToTargetConverter,
                targetToSourceConverter, pixelConverter)
        {
            SkipRecords = skipCount;
        }

        protected int SkipRecords { get; set; }

        public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels)
        {
            List<string> files =
                Directory.EnumerateFiles(DirectoryPath,
                    string.Join("|", ValidImageExtensions.Select(a => string.Format("*{0}", a))),
                    SearchOption.AllDirectories)
                    .Skip(SkipRecords)
                    .Take(count)
                    .ToList();

            labelsEncoded = new T[files.Count, LabelDataWidth];
            labels = new string[files.Count];
            var data = new T[files.Count, DataWidth];
            var on = (T) Convert.ChangeType(1, typeof (T));
            var off = (T) Convert.ChangeType(0, typeof (T));
            for (int i = 0; i < files.Count; i++)
            {
                string lblName = new FileInfo(files[i]).Directory.Name;
                labels[i] = lblName;
                if (UseGrayCodesForLabels)
                {
                    GrayLabelEncoder.Encode(lblName, labelsEncoded, i, 0, on, off);
                }
                else
                {
                    labelsEncoded[i, NonGrayEncoderIndexes[lblName]] = on;
                }

                CopyImageDataToTarget(data, i, 0, files[i]);
            }

            return data;
        }


        public override T[,] Read(int count)
        {
            List<string> files =
                Directory.EnumerateFiles(DirectoryPath,
                    string.Join("|", ValidImageExtensions.Select(a => string.Format("*{0}", a))),
                    SearchOption.AllDirectories)
                    .Skip(SkipRecords)
                    .Take(count)
                    .ToList();

            var data = new T[files.Count, DataWidth];
            var on = (T) Convert.ChangeType(1, typeof (T));
            var off = (T) Convert.ChangeType(0, typeof (T));
            for (int i = 0; i < files.Count; i++)
            {
                CopyImageDataToTarget(data, i, 0, files[i]);
            }
            return data;
        }


        public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded,
            out IList<string[]> labels)
        {
            var res = new List<T[,]>();
            var lbls = new List<string[]>();
            var coded = new List<T[,]>();
            var on = (T) Convert.ChangeType(1, typeof (T));
            var off = (T) Convert.ChangeType(0, typeof (T));

            List<string> files =
                Directory.EnumerateFiles(DirectoryPath,
                    string.Join("|", ValidImageExtensions.Select(a => string.Format("*{0}", a))),
                    SearchOption.AllDirectories)
                    .Skip(SkipRecords)
                    .Take(count)
                    .ToList();

            IEnumerable<IList<string>> batches = Partition(files, batchSize);
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
            var on = (T) Convert.ChangeType(1, typeof (T));
            var off = (T) Convert.ChangeType(0, typeof (T));

            List<string> files =
                Directory.EnumerateFiles(DirectoryPath,
                    string.Join("|", ValidImageExtensions.Select(a => string.Format("*{0}", a))),
                    SearchOption.AllDirectories)
                    .Skip(SkipRecords)
                    .Take(count)
                    .ToList();

            IEnumerable<IList<string>> batches = Partition(files, batchSize);
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

        //public override T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}

        //public override T[,] Read(int count, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}

        //public override IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded, out IList<string[]> labels, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}

        //public override IList<T[,]> Read(int count, int batchSize, Func<T, T> sourceToTargetConverter)
        //{
        //    throw new NotImplementedException();
        //}
    }
}