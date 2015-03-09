using System;
using System.Collections.Generic;
using SimpleRBM.Demo.Util;

namespace CudaNN.DeepBelief.DataIO
{
    public class OneOfNOptionsGrayCodedFieldReader<T> : FieldReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        private readonly FieldGrayEncoder<string> _graysCoder;
        private readonly T _offValue;
        private readonly T _onValue;

        public OneOfNOptionsGrayCodedFieldReader(int sourceIndex, int targetIndex, IEnumerable<string> options,
            T onValue, T offValue, Func<T, T> sourceToTarget,
            Func<T, T> targetToSource)
            : base(sourceIndex, targetIndex, null, sourceToTarget, targetToSource)
        {
            _graysCoder = new FieldGrayEncoder<string>(options);
            _onValue = onValue;
            _offValue = offValue;
        }

        public override int TargetWidth
        {
            get { return _graysCoder.ElementsRequired; }
        }

        public override void CopyToTarget(T[,] target, string[] line, int targetRow)
        {
            CopyToTarget(target, line, targetRow, SourceToTargetConverter);
        }

        public override void ReadFromTarget(T[,] target, string[] line, int targetRow)
        {
            throw new NotImplementedException();
        }

        public override string ReadValueFromTarget(T[,] target, int targetRow)
        {
            return ReadValueFromTarget(target, targetRow, TargetToSourceConverter);
        }

        public override void CopyToTarget(T[,] target, string[] line, int targetRow, Func<T, T> sourceToTargetConverter)
        {
            _graysCoder.Encode(line[SourceIndex], target, targetRow, TargetIndex, sourceToTargetConverter(_onValue), sourceToTargetConverter(_offValue));

        }

        public override void ReadFromTarget(T[,] target, string[] line, int targetRow, Func<T, T> targetToSourceConverter)
        {
            throw new NotImplementedException();
        }

        public override string ReadValueFromTarget(T[,] target, int targetRow, Func<T, T> targetToSourceConverter)
        {
            return _graysCoder.Decode(target, targetRow, TargetIndex, targetToSourceConverter(_onValue), targetToSourceConverter(_offValue));
        }
    }
}