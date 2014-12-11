using System;

namespace SimpleRBM.Common.ExitCondition
{
    public class EpochCountExitConditionFactory<T> : IExitConditionEvaluatorFactory<T> where T : struct, IComparable<T>
    {
        private readonly int _maxEpochs;

        public EpochCountExitConditionFactory(int maxEpochs)
        {
            Console.WriteLine("RBMs will exit after {0} epochs", maxEpochs);
            _maxEpochs = maxEpochs;
        }

        public IExitConditionEvaluator<T> Create(int layerDepth, int inputNodes, int outputNodes)
        {
            return new EpochCountExitCondition<T>()
            {
                MaxEpoch = _maxEpochs
            };
        }
    }
}