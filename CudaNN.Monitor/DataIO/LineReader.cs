using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CudaNN.DeepBelief.DataIO
{
    public class LineReader<T> where T : IComparable<T>, IEquatable<T>
    {
        private readonly FieldReaderBase<T>[] _fieldReaders;

        public int DataWidth { get; protected set; }

        public LineReader(IEnumerable<FieldReaderBase<T>> fieldReaders)
        {
            _fieldReaders = fieldReaders.ToArray();
            DataWidth = _fieldReaders.Sum(a => a.TargetWidth);
        }

        public void CopyToTarget(T[,] target, string[] line, int targetRow)
        {
            Parallel.ForEach(_fieldReaders, a => a.CopyToTarget(target, line, targetRow));
        }

        public void ReadFromTarget(T[,] target, string[] line, int targetRow)
        {
            Parallel.ForEach(_fieldReaders, a => a.ReadFromTarget(target, line, targetRow));
        }

        public IEnumerable<string> ReadTargetLineContents(T[,] target, int row)
        {
            return _fieldReaders.Select(t => t.ReadValueFromTarget(target, row));
        }

        public void CopyToTarget(T[,] target, string[] line, int targetRow, Func<T, T> sourceToTargetConverter)
        {
            Parallel.ForEach(_fieldReaders, a => a.CopyToTarget(target, line, targetRow, sourceToTargetConverter));
        }

        public void ReadFromTarget(T[,] target, string[] line, int targetRow, Func<T, T> targetToSourceConverter)
        {
            Parallel.ForEach(_fieldReaders, a => a.ReadFromTarget(target, line, targetRow, targetToSourceConverter));
        }

        public IEnumerable<string> ReadTargetLineContents(T[,] target, int row, Func<T, T> targetToSourceConverter)
        {
            return _fieldReaders.Select(t => t.ReadValueFromTarget(target, row, targetToSourceConverter));
        }
    }
}