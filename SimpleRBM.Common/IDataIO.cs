using System;

namespace SimpleRBM.Common
{
    public interface IDataIO<TDataElement, TLabel> where TDataElement : struct, IComparable<TDataElement>
    {
        TDataElement[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels);
        TDataElement[,] ReadTestData(int skipRecords, int count);

        void PrintToScreen(TDataElement[,] arr, TLabel[] labels = null, TDataElement[,] reference = null,
            ulong[] keys = null);

        void PrintMap(TDataElement[,] arr);
    }
}