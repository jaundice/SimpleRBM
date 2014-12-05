using System;

namespace SimpleRBM.Common
{
    public interface IDataIO<T> where T : struct, IComparable<T>
    {
        T[,] ReadTrainingData(string filePath, int startLine, int count, out int[] labels);
        T[,] ReadTestData(string filePath, int startLine, int count);

        void PrintToScreen(T[,] arr, int[] labels = null, T[,] reference = null, ulong[] keys = null);
        void PrintMap(T[,] arr);
    }
}