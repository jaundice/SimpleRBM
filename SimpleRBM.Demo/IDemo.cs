using System;
using SimpleRBM.Common;

namespace SimpleRBM.Demo
{
    internal interface IDemo
    {
        void Execute<T>(IDeepBeliefNetworkFactory<T> dbnFactory,
            IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, int[] defaultLayerSizes,
            IDataIO<T> dataProvider, T learningRate, int trainingSize, int skipTrainingRecords)
            where T : struct,
                IComparable<T>;
    }
}