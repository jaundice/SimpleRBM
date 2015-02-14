namespace SimpleRBM.Common.LearningRate
{
    public class LinearlyDecayingLearningRateFactory<T> : ILearningRateCalculatorFactory<T>
    {
        private readonly double _decayRate;
        private readonly double _initialRate;
        private double _minRate;

        public LinearlyDecayingLearningRateFactory(double initialRate, double decayRate, double minRate)
        {
            _initialRate = initialRate;
            _decayRate = decayRate;
            _minRate = minRate;
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return new LinearlyDecayingLearningRate<T>(_initialRate, _decayRate, _minRate);
        }
    }
}