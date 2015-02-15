using System;

namespace SimpleRBM.Common.ExitCondition
{
    public class EpochCountExitConditionFactory<T> : IExitConditionEvaluatorFactory<T> where T : struct, IComparable<T>
    {
        private readonly int _maxEpochs;
        private readonly IEpochErrorTracker<T> _epochErrorTracker;

        public EpochCountExitConditionFactory(IEpochErrorTracker<T> epochErrorTracker, int maxEpochs)
        {
            Console.WriteLine("RBMs will exit after {0} epochs", maxEpochs);
            _epochErrorTracker = epochErrorTracker;
            _maxEpochs = maxEpochs;
        }

        public IExitConditionEvaluator<T> Create(int layerIndex)
        {
            return new EpochCountExitCondition<T>(_epochErrorTracker)
            {
                LayerDepth = layerIndex,
                MaxEpoch = _maxEpochs
            };
        }
    }
}