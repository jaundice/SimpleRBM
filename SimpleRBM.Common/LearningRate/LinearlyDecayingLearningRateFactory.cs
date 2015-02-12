namespace SimpleRBM.Common.LearningRate
{
    public class LinearlyDecayingLearningRateFactory<T> : ILearningRateCalculatorFactory<T>
    {
        private readonly double _decayRate;
        private readonly double _initialRate;

        public LinearlyDecayingLearningRateFactory(double initialRate, double decayRate)
        {
            _initialRate = initialRate;
            _decayRate = decayRate;
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return new LinearlyDecayingLearningRate<T>(_initialRate, _decayRate);
        }
    }
}