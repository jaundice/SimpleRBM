using System;

namespace SimpleRBM.Common.ExitCondition
{
    public class ManualKeyPressExitEvaluatorFactory<T> : IExitConditionEvaluatorFactory<T> where T : struct, IComparable<T>
    {
        private readonly int _maxEpochs;
        private readonly T _minError;

        public ManualKeyPressExitEvaluatorFactory(T minError, int maxEpochs = 100000)
        {
            _maxEpochs = maxEpochs;
            _minError = minError;
        }

        public IExitConditionEvaluator<T> Create(int layerDepth, int inputNodes, int outputNodes)
        {
            return new ManualKeyPressEvaluator<T>(_maxEpochs, _minError);
        }
    }
}