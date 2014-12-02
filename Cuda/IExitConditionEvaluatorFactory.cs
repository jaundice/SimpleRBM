using System;

namespace CudaRbm
{
    public interface IExitConditionEvaluatorFactory<T> where T : IComparable<T>
    {
        IExitConditionEvaluator<T> Create(int layerDepth, int inputNodes, int outputNodes);
    }
}