using System;
using System.Threading.Tasks;

namespace CudaNN.DeepBelief.DataIO
{
    public abstract class TextFileReaderBase<T> : DataReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        protected TextFileReaderBase(LineReader<T> labelReader, LineReader<T> dataReader, bool firstLineIsHeaders,
            int totalRecordCount, string filePath, char fieldSeparator)
           
        {
            LabelReader = labelReader;
            DataReader = dataReader;
            FirstLineIsHeaders = firstLineIsHeaders;
            FilePath = filePath;
            TotalRecordCount = totalRecordCount;
            FieldSeparator = fieldSeparator;
        }

        protected LineReader<T> LabelReader { get; set; }
        protected LineReader<T> DataReader { get; set; }
        protected bool FirstLineIsHeaders { get; set; }
        protected string FilePath { get; set; }
        protected char FieldSeparator { get; set; }

        public override int LabelDataWidth
        {
            get { return LabelReader == null ? 0 : LabelReader.DataWidth; }
        }

        public override int DataWidth
        {
            get { return DataReader.DataWidth; }
        }

        public override string[] DecodeLabels(T[,] llbl, T onValue, T offValue)
        {
            var ret = new string[llbl.GetLength(0)];
            Parallel.For(0, ret.GetLength(0),
                i => { ret[i] = string.Join(",", LabelReader.ReadTargetLineContents(llbl, i)); });

            return ret;
        }
    }
}