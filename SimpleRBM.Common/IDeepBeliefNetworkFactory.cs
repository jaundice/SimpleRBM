using System;
using System.IO;

namespace SimpleRBM.Common
{
    public interface IDeepBeliefNetworkFactory<T> where T : struct, IComparable<T>
    {
        IDeepBeliefNetwork<T> Create(DirectoryInfo networkDataDir, int[] appendLayers = null,
            T learningRate = default (T), IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory = null);


        IDeepBeliefNetwork<T> Create(int[] layerSizes = null, T learningRate = default (T),
            IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory = null);
    }
}