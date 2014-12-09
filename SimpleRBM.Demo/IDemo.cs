using System;
using SimpleRBM.Common;

namespace SimpleRBM.Demo
{
    internal interface IDemo
    {
        void Execute<TDataElement,TLabel>(IDeepBeliefNetworkFactory<TDataElement> dbnFactory,
            IExitConditionEvaluatorFactory<TDataElement> exitConditionEvaluatorFactory, int[] defaultLayerSizes,
            IDataIO<TDataElement,TLabel> dataProvider, TDataElement learningRate, int trainingSize, int skipTrainingRecords)
            where TDataElement : struct,
                IComparable<TDataElement>;
    }
}