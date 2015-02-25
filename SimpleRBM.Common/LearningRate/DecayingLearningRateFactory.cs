namespace SimpleRBM.Common.LearningRate
{
    public class DecayingLearningRateFactory<T> : ILearningRateCalculatorFactory<T>
    {
        private readonly double _decayRate;
        private readonly double _initialRate;
        private double _minRate;
        private DecayingLearningRate<T>.Decay _decayFunc;

        public DecayingLearningRateFactory(double initialRate, double decayRate, double minRate,
            DecayType decayType = DecayType.Power)
            : this(initialRate, decayRate, minRate, DecayingLearningRate<T>.GetDecay(decayType))
        {
        }

        public DecayingLearningRateFactory(double initialRate, double decayRate, double minRate,
            DecayingLearningRate<T>.Decay decayFunc)
        {
            _initialRate = initialRate;
            _decayRate = decayRate;
            _minRate = minRate;
            _decayFunc = decayFunc;
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return new DecayingLearningRate<T>(_initialRate, _decayRate, _minRate, _decayFunc);
        }
    }
}