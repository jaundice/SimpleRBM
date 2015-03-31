using System;
using System.Collections.Generic;

namespace CudaNN.DeepBelief.DataIO
{
    public abstract class DataReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {

        protected int TotalRecordCount { get; set; }
        public abstract int LabelDataWidth { get; }

        public abstract int DataWidth { get; }
        public abstract T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels);
        public abstract T[,] Read(int count);

        public abstract IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded, out IList<string[]> labels);
        public abstract IList<T[,]> Read(int count, int batchSize);


        //public abstract T[,] ReadWithLabels(int count, out T[,] labelsEncoded, out string[] labels, Func<T, T> sourceToTargetConverter);
        //public abstract T[,] Read(int count, Func<T, T> sourceToTargetConverter);

        //public abstract IList<T[,]> ReadWithLabels(int count, int batchSize, out IList<T[,]> labelsEncoded, out IList<string[]> labels, Func<T, T> sourceToTargetConverter);
        //public abstract IList<T[,]> Read(int count, int batchSize, Func<T, T> sourceToTargetConverter);


        public abstract string[] DecodeLabels(T[,] llbl, T onValue, T offValue);
    }
}