using System;
using SimpleRBM.Common;

namespace SimpleRBM.Demo
{
    internal interface IDemo
    {
        void Execute<TDataElement, TLabel>(IDeepBeliefNetworkFactory<TDataElement> dbnFactory,
            IExitConditionEvaluatorFactory<TDataElement> exitConditionEvaluatorFactory, ILayerDefinition[] defaultLayerSizes,
            IDataIO<TDataElement, TLabel> dataProvider, ILearningRateCalculator<TDataElement> learningRateCalculator, int trainingSize, int skipTrainingRecords, bool classify = true)
            where TDataElement : struct,
                IComparable<TDataElement>;
    }
}