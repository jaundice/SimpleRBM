namespace SimpleRBM.Common.LearningRate
{
    public class DecayingLearningRateFactory<T> : ILearningRateCalculatorFactory<T>
    {
        private readonly double _decayRate;
        private readonly double _initialRate;
        private readonly double _minRate;
        private readonly DecayingLearningRate<T>.Decay _decayFunc;
        private readonly int _notifyFreq;

        public DecayingLearningRateFactory(double initialRate, double decayRate, double minRate,
            DecayType decayType = DecayType.Power, int notifyFreq=-1)
            : this(initialRate, decayRate, minRate, DecayingLearningRate<T>.GetDecay(decayType), notifyFreq)
        {
        }

        public DecayingLearningRateFactory(double initialRate, double decayRate, double minRate,
            DecayingLearningRate<T>.Decay decayFunc, int notifyFreq)
        {
            _initialRate = initialRate;
            _decayRate = decayRate;
            _minRate = minRate;
            _decayFunc = decayFunc;
            _notifyFreq = notifyFreq;
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return new DecayingLearningRate<T>(_initialRate, _decayRate, _minRate, _decayFunc, _notifyFreq);
        }
    }
}