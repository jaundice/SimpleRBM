using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Demo;
using SimpleRBM.Demo.Util;

namespace CudaNN.DeepBelief.DataIO
{
    public abstract class ImageReaderBase<T> : DataReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        private readonly int _dataWidth;

        protected ImageReaderBase(string directoryPath, bool useGrayCodesForLabels, int dataWidth,
            IEnumerable<string> allLabels, string[] imageExtensions, int totalRecordCount, ImageUtils.ConvertPixel<T> convertFromImage,
            Func<T, byte> convertToImage)
        {
            DirectoryPath = directoryPath;
            UseGrayCodesForLabels = useGrayCodesForLabels;
            ConvertFromImage = convertFromImage;
            ConvertToImage = convertToImage;
            ValidImageExtensions = imageExtensions;
            TotalRecordCount = totalRecordCount;
            _dataWidth = dataWidth;
            var hs = new HashSet<string>(allLabels);

            NonGrayEncoderIndexes = hs.OrderBy(a => a)
                .Select((a, i) => new { ind = i, el = a })
                .ToDictionary(a => a.el, a => a.ind);

            NonGrayEncoderInverseIndexes = NonGrayEncoderIndexes.ToDictionary(a => a.Value, a => a.Key);

            GrayLabelEncoder = new FieldGrayEncoder<string>(hs);
        }

        public Dictionary<int, string> NonGrayEncoderInverseIndexes { get; set; }

        public override string[] DecodeLabels(T[,] llbl, T onValue, T offValue)
        {
            string[] ret = new string[llbl.GetLength(0)];

            if (UseGrayCodesForLabels)
            {
                Parallel.For(0, ret.GetLength(0), a => ret[a] = GrayLabelEncoder.Decode(llbl, a, 0, onValue, offValue));
            }
            else
            {
                Parallel.For(0, ret.GetLength(0), i =>
                {
                    List<string> opts = new List<string>();
                    for (int j = 0; j < llbl.GetLength(1); j++)
                    {
                        if (Comparer<T>.Default.Compare(llbl[i, j], onValue) == 0)
                        {
                            opts.Add(NonGrayEncoderInverseIndexes[j]);
                        }
                    }
                    ret[i] = string.Join(",", opts);
                });
            }
            return ret;
        }

        protected string DirectoryPath { get; set; }
        protected bool UseGrayCodesForLabels { get; set; }
        protected ImageUtils.ConvertPixel<T> ConvertFromImage { get; set; }
        protected Func<T, byte> ConvertToImage { get; set; }
        protected FieldGrayEncoder<string> GrayLabelEncoder { get; set; }
        protected Dictionary<string, int> NonGrayEncoderIndexes { get; set; }
        protected string[] ValidImageExtensions { get; set; }

        public override int LabelDataWidth
        {
            get { return UseGrayCodesForLabels ? GrayLabelEncoder.ElementsRequired : NonGrayEncoderIndexes.Count; }
        }

        public override int DataWidth
        {
            get { return _dataWidth; }
        }

        protected void CopyImageDataToTarget(T[,] target, int targetRow, int rowOffset, string filePath)
        {
            ImageUtils.CopyImageDataTo(filePath, target, targetRow, rowOffset, ConvertFromImage);
        }

        protected IEnumerable<IList<T>> Partition<T>(List<T> files, int batchSize)
        {
            List<T> batch = new List<T>(batchSize);
            var enu = files.GetEnumerator();
            while (enu.MoveNext())
            {
                batch.Add(enu.Current);
                if (batch.Count == batchSize)
                {
                    yield return batch;
                    batch = new List<T>();
                }
            }
            if (batch.Count > 0)
                yield return batch;

        }
    }
}