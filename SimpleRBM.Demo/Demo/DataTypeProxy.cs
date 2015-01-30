using SimpleRBM.Common;

namespace SimpleRBM.Demo.Demo
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

        void IDataIO<double, TLabel>.PrintToScreen(double[,] arr, double[,] reference = null, TLabel[] referenceLabels = null, double[,] referenceLabelsCoded = null, ulong[][] keys = null, double[,] computedLabels = null)
        {
            _doubleProvider.PrintToScreen(arr, reference: reference, referenceLabels: referenceLabels, referenceLabelsCoded: referenceLabelsCoded, keys: keys, computedLabels:computedLabels);
        }

        void IDataIO<double, TLabel>.PrintMap(double[,] arr)
        {
            _doubleProvider.PrintMap(arr);
        }

        float[,] IDataIO<float, TLabel>.ReadTrainingData(int skipRecords, int count, out TLabel[] labels, out float[,] labelsCoded)
        {
            return _floatProvider.ReadTrainingData(skipRecords, count, out labels, out labelsCoded);
        }

        float[,] IDataIO<float, TLabel>.ReadTestData(int skipRecords, int count)
        {
            return _floatProvider.ReadTestData(skipRecords, count);
        }

        void IDataIO<float, TLabel>.PrintToScreen(float[,] arr, float[,] reference = null, TLabel[] referenceLabels = null, float[,] referenceLabelsCoded = null, ulong[][] keys = null, float[,] computedLabels = null)
        {
            _floatProvider.PrintToScreen(arr, reference: reference, referenceLabels: referenceLabels, referenceLabelsCoded: referenceLabelsCoded, keys: keys, computedLabels:computedLabels);
        }

        void IDataIO<float, TLabel>.PrintMap(float[,] arr)
        {
            _floatProvider.PrintMap(arr);
        }
    }
}