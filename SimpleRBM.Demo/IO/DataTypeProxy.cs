using SimpleRBM.Common;

namespace SimpleRBM.Demo.IO
{
    public class IODataTypeProxy<TLabel> : IDataIO<float, TLabel>, IDataIO<double, TLabel>
    {
        private readonly IDataIO<double, TLabel> _doubleProvider;
        private readonly IDataIO<float, TLabel> _floatProvider;

        public IODataTypeProxy(IDataIO<float, TLabel> floatProvider, IDataIO<double, TLabel> doubleProvider)
        {
            _floatProvider = floatProvider;
            _doubleProvider = doubleProvider;
        }


        double[,] IDataIO<double, TLabel>.ReadTrainingData(int skipRecords, int count, out TLabel[] labels, out double[,] labelsCoded)
        {
            return _doubleProvider.ReadTrainingData(skipRecords, count, out labels, out labelsCoded);
        }

        double[,] IDataIO<double, TLabel>.ReadTestData(int skipRecords, int count)
        {
            return _doubleProvider.ReadTestData(skipRecords, count);
        }

        void IDataIO<double, TLabel>.PrintToConsole(double[,] arr, double[,] reference = null, TLabel[] referenceLabels = null, double[,] referenceLabelsCoded = null, ulong[][] keys = null, double[,] computedLabels = null)
        {
            _doubleProvider.PrintToConsole(arr, reference: reference, referenceLabels: referenceLabels, referenceLabelsCoded: referenceLabelsCoded, keys: keys, computedLabels:computedLabels);
        }

        void IDataIO<double, TLabel>.PrintToConsole(double[,] arr)
        {
            _doubleProvider.PrintToConsole(arr);
        }

        float[,] IDataIO<float, TLabel>.ReadTrainingData(int skipRecords, int count, out TLabel[] labels, out float[,] labelsCoded)
        {
            return _floatProvider.ReadTrainingData(skipRecords, count, out labels, out labelsCoded);
        }

        float[,] IDataIO<float, TLabel>.ReadTestData(int skipRecords, int count)
        {
            return _floatProvider.ReadTestData(skipRecords, count);
        }

        void IDataIO<float, TLabel>.PrintToConsole(float[,] arr, float[,] reference = null, TLabel[] referenceLabels = null, float[,] referenceLabelsCoded = null, ulong[][] keys = null, float[,] computedLabels = null)
        {
            _floatProvider.PrintToConsole(arr, reference: reference, referenceLabels: referenceLabels, referenceLabelsCoded: referenceLabelsCoded, keys: keys, computedLabels:computedLabels);
        }

        void IDataIO<float, TLabel>.PrintToConsole(float[,] arr)
        {
            _floatProvider.PrintToConsole(arr);
        }
    }
}