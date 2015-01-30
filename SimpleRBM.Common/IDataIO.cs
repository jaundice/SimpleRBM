using System;

namespace SimpleRBM.Common
{
    public interface IDataIO<TDataElement, TLabel> where TDataElement : struct, IComparable<TDataElement>
    {
        TDataElement[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels,
            out TDataElement[,] labelsCoded);

        TDataElement[,] ReadTestData(int skipRecords, int count);

        void PrintToScreen(TDataElement[,] arr, TDataElement[,] reference = null, TLabel[] referenceLabels = null,
            TDataElement[,] referenceLabelsCoded = null, ulong[][] keys = null, TDataElement[,] computedLabels = null);

        void PrintMap(TDataElement[,] arr);
    }
}