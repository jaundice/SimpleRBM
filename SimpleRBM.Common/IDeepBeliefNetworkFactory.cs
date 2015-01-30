using System;
using System.IO;

namespace SimpleRBM.Common
{
    public interface IDeepBeliefNetworkFactory<T> where T : struct, IComparable<T>
    {
        IDeepBeliefNetwork<T> Create(DirectoryInfo networkDataDir, ILayerDefinition[] appendLayers,
            ILearningRateCalculator<T> learningRateCalculator , IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory = null);


        IDeepBeliefNetwork<T> Create(ILayerDefinition[] layerSizes, ILearningRateCalculator<T> learningRateCalculator,
            IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory = null);
    }
}