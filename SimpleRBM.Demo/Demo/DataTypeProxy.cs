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


        double[,] IDataIO<double, TLabel>.ReadTrainingData(int skipRecords, int count, out TLabel[] labels)
        {
            return _doubleProvider.ReadTrainingData(skipRecords, count, out labels);
        }

        double[,] IDataIO<double, TLabel>.ReadTestData(int skipRecords, int count)
        {
            return _doubleProvider.ReadTestData(skipRecords, count);
        }

        void IDataIO<double, TLabel>.PrintToScreen(double[,] arr, TLabel[] labels, double[,] reference, ulong[] keys)
        {
            _doubleProvider.PrintToScreen(arr, labels, reference, keys);
        }

        void IDataIO<double, TLabel>.PrintMap(double[,] arr)
        {
            _doubleProvider.PrintMap(arr);
        }

        float[,] IDataIO<float, TLabel>.ReadTrainingData(int skipRecords, int count, out TLabel[] labels)
        {
            return _floatProvider.ReadTrainingData(skipRecords, count, out labels);
        }

        float[,] IDataIO<float, TLabel>.ReadTestData(int skipRecords, int count)
        {
            return _floatProvider.ReadTestData(skipRecords, count);
        }

        void IDataIO<float, TLabel>.PrintToScreen(float[,] arr, TLabel[] labels, float[,] reference, ulong[] keys)
        {
            _floatProvider.PrintToScreen(arr, labels, reference, keys);
        }

        void IDataIO<float, TLabel>.PrintMap(float[,] arr)
        {
            _floatProvider.PrintMap(arr);
        }
    }
}