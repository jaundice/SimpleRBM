using System;

namespace CudaNN.DeepBelief.DataIO
{
    public abstract class FieldReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        protected FieldReaderBase(int sourceIndex, int targetIndex, Func<string, T> parser, Func<T,T> sourceToTargetConverter, Func<T,T> targetToSourceConverter )
        {
            SourceIndex = sourceIndex;
            TargetIndex = targetIndex;
            Parser = parser;
            SourceToTargetConverter = sourceToTargetConverter;
            TargetToSourceConverter = targetToSourceConverter;
        }

        public Func<T, T> SourceToTargetConverter { get; protected set; }
        public Func<T, T> TargetToSourceConverter { get; protected set; }



        public int SourceIndex { get; protected set; }
        public int TargetIndex { get; protected set; }
        public Func<string, T> Parser { get; protected set; }
        public abstract int TargetWidth { get; }
        public abstract void CopyToTarget(T[,] target, string[] line, int targetRow);
        public abstract void ReadFromTarget(T[,] target, string[] line, int targetRow);
        public abstract string ReadValueFromTarget(T[,] target, int targetRow);

        public abstract void CopyToTarget(T[,] target, string[] line, int targetRow, Func<T, T> sourceToTargetConverter);
        public abstract void ReadFromTarget(T[,] target, string[] line, int targetRow, Func<T, T> targetToSourceConverter);
        public abstract string ReadValueFromTarget(T[,] target, int targetRow, Func<T, T> targetToSourceConverter);


    }
}