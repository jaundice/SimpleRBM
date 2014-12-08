using System;

namespace SimpleRBM.Common
{
    public interface IDataIO<T,L> where T : struct, IComparable<T>
    {
        T[,] ReadTrainingData(string filePath, int startLine, int count, out L[] labels);
        T[,] ReadTestData(string filePath, int startLine, int count);

        void PrintToScreen(T[,] arr, L[] labels = null, T[,] reference = null, ulong[] keys = null);
        void PrintMap(T[,] arr);
    }
}