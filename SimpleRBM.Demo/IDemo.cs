using System;
using SimpleRBM.Common;

namespace SimpleRBM.Demo
{
    internal interface IDemo
    {
        void Execute<TDataElement, TLabel>(IDeepBeliefNetworkFactory<TDataElement> dbnFactory,
            ILayerDefinition[] defaultLayerSizes, IDataIO<TDataElement, TLabel> dataProvider,
            ILearningRateCalculatorFactory<TDataElement> preTrainLearningRateCalculatorFactory,
            IExitConditionEvaluatorFactory<TDataElement> preTrainExitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TDataElement> fineTrainLearningRateCalculatorFactory,
            IExitConditionEvaluatorFactory<TDataElement> fineTrainExitConditionEvaluatorFactory, int trainingSize,
            int skipTrainingRecords, bool classify = true)
            where TDataElement : struct,
                IComparable<TDataElement>;
    }
}