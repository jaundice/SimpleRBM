using System;
using System.Collections.Generic;

namespace SimpleRBM.Common
{
    public interface IDataIO<TDataElement, TLabel> where TDataElement : struct, IComparable<TDataElement>
    {
        TDataElement[,] ReadTrainingData(int skipRecords, int count, out TLabel[] labels,
            out TDataElement[,] labelsCoded);

        IList<TDataElement[,]> ReadTrainingData(int skipRecords, int count, int batchSize, out IList<TLabel[]> labels,
           out IList<TDataElement[,]> labelsCoded);

        TDataElement[,] ReadTestData(int skipRecords, int count);

        IList<TDataElement[,]> ReadTestData(int skipRecords, int count, int batchSize);

        void PrintToConsole(TDataElement[,] arr, TDataElement[,] reference = null, TLabel[] referenceLabels = null,
            TDataElement[,] referenceLabelsCoded = null, ulong[][] keys = null, TDataElement[,] computedLabels = null);

        void PrintToConsole(TDataElement[,] arr);
    }
}