using System;

namespace CudaNN.DeepBelief.DataIO
{
    public class RealFieldReader<T> : FieldReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        public RealFieldReader(int sourceIndex, int targetIndex, Func<string, T> parser)
            : this(sourceIndex, targetIndex, parser, a => a, a => a)
        {
        }

        public RealFieldReader(int sourceIndex, int targetIndex, Func<string, T> parser, Func<T, T> sourceToTarget,
            Func<T, T> targetToSource)
            : base(sourceIndex, targetIndex, parser, sourceToTarget, targetToSource)
        {

        }

        public override int TargetWidth
        {
            get { return 1; }
        }

        public override void CopyToTarget(T[,] target, string[] line, int targetRow)
        {
            CopyToTarget(target, line, targetRow, SourceToTargetConverter);
        }

        public override void ReadFromTarget(T[,] target, string[] line, int targetRow)
        {
            ReadFromTarget(target, line, targetRow, TargetToSourceConverter);
        }

        public override string ReadValueFromTarget(T[,] target, int targetRow)
        {
            return ReadValueFromTarget(target, targetRow, TargetToSourceConverter);
        }

        public override void CopyToTarget(T[,] target, string[] line, int targetRow, Func<T, T> sourceToTargetConverter)
        {
            var s = line[SourceIndex];
            if (!string.IsNullOrWhiteSpace(s))
            {
                T parsed = Parser(s);
                T converted = sourceToTargetConverter(parsed);
                target[targetRow, TargetIndex] = converted;
            }
        }

        public override void ReadFromTarget(T[,] target, string[] line, int targetRow, Func<T, T> targetToSourceConverter)
        {
            line[SourceIndex] = ReadValueFromTarget(target, targetRow, targetToSourceConverter);
        }

        public override string ReadValueFromTarget(T[,] target, int targetRow, Func<T, T> targetToSourceConverter)
        {
            return targetToSourceConverter(target[targetRow, TargetIndex]).ToString();
        }
    }
}