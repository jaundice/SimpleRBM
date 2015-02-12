using System;

namespace SimpleRBM.Common.ExitCondition
{
    public class CompanionDatasetExitConditionEvaluatorFactory<T> : IExitConditionEvaluatorFactory<T> where T : struct, IComparable<T>
    {
        private readonly int _updateEpochs;
        private readonly int _maxEpochs;
        public IDeepBeliefNetworkExtended<T> Dbn { get; set; }
        public T[,] TestData { get; set; }

        public CompanionDatasetExitConditionEvaluatorFactory(IDeepBeliefNetworkExtended<T> dbn, int maxEpochs, int updateEpochs, IEpochErrorTracker<T> mainTracker, IEpochErrorTracker<T> companionTracker)
        {
            _updateEpochs = updateEpochs;
            _maxEpochs = maxEpochs;
            Dbn = dbn;
            MainTracker = mainTracker;
            CompanionTracker = companionTracker;
        }

        public IEpochErrorTracker<T> CompanionTracker { get; protected set; }

        public IEpochErrorTracker<T> MainTracker { get; protected set; }

        public IExitConditionEvaluator<T> Create(int layerIndex)
        {
            return new CompanionDatasetExitConditionEvaluator<T>(this, layerIndex, _updateEpochs, _maxEpochs);
        }
    }
}