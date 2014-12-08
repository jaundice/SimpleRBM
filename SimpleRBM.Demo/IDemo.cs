using System;
using SimpleRBM.Common;

namespace SimpleRBM.Demo
{
    internal interface IDemo
    {
        void Execute<T,L>(IDeepBeliefNetworkFactory<T> dbnFactory,
            IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, int[] defaultLayerSizes,
            IDataIO<T,L> dataProvider, T learningRate, int trainingSize, int skipTrainingRecords)
            where T : struct,
                IComparable<T>;
    }
}