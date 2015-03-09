using System;
using System.Collections.Generic;
using System.Linq;
using Mono.CSharp;

namespace CudaNN.DeepBelief.DataIO
{
    public class OneOfNOptionsFieldReader<T> : FieldReaderBase<T> where T : IComparable<T>, IEquatable<T>
    {
        private readonly Dictionary<int, string> _inverseMap;
        private readonly Dictionary<string, int> _nameToIndexMap;
        private readonly T _onValue;
        private T _offValue;

        public OneOfNOptionsFieldReader(int sourceIndex, int targetIndex, IEnumerable<string> options, T onValue,
            T offValue, Func<T, T> sourceToTarget,
            Func<T, T> targetToSource)
            : base(sourceIndex, targetIndex, null, sourceToTarget, targetToSource)
        {
            _nameToIndexMap = options.OrderBy(a => a)
                .Select((a, i) => new { ind = i, key = a })
                .ToDictionary(a => a.key, a => a.ind + TargetIndex);
            _inverseMap = _nameToIndexMap.ToDictionary(a => a.Value, a => a.Key);
            _onValue = onValue;
            _offValue = offValue;
        }

        public override int TargetWidth
        {
            get { return _nameToIndexMap.Count; }
        }

        public override void CopyToTarget(T[,] target, string[] line, int targetRow)
        {
            CopyToTarget(target, line, targetRow, SourceToTargetConverter);
        }

        public override void ReadFromTarget(T[,] target, string[] line, int targetRow)
        {
            ReadValueFromTarget(target, targetRow, TargetToSourceConverter);
        }

        public override string ReadValueFromTarget(T[,] target, int targetRow)
        {

            return ReadValueFromTarget(target, targetRow, TargetToSourceConverter);
        }

        public override void CopyToTarget(T[,] target, string[] line, int targetRow, Func<T, T> sourceToTargetConverter)
        {
            target[targetRow, _nameToIndexMap[line[SourceIndex]]] = sourceToTargetConverter(_onValue);
        }

        public override void ReadFromTarget(T[,] target, string[] line, int targetRow, Func<T, T> targetToSourceConverter)
        {
            for (int i = TargetIndex; i < TargetIndex + TargetWidth; i++)
            {
                if (Comparer<T>.Default.Compare(targetToSourceConverter(target[targetRow, i]), _onValue) == 0)
                //may need enhancing to cope with non exact results
                {
                    line[SourceIndex] = _inverseMap[i];
                    break;
                }
            }
        }

        public override string ReadValueFromTarget(T[,] target, int targetRow, Func<T, T> targetToSourceConverter)
        {
            //todo handle non exact results;
            List<string> options = new List<string>();
            for (var i = TargetIndex; i < TargetIndex + _inverseMap.Count; i++)
            {
                if (Comparer<T>.Default.Compare(targetToSourceConverter(target[targetRow, i]), _onValue) == 0)
                {
                    options.Add(_inverseMap[i]);
                }
            }

            return string.Join(",", options);
        }
    }
}