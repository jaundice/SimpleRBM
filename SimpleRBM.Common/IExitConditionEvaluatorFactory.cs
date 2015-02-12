using System;

namespace SimpleRBM.Common
{
    public interface IExitConditionEvaluatorFactory<T> where T : struct, IComparable<T>
    {
        IExitConditionEvaluator<T> Create(int layerIndex);
    }
}