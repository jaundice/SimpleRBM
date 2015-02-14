using System;

namespace SimpleRBM.Common.LearningRate
{
    public class LinearlyDecayingLearningRate<T> : ILearningRateCalculator<T>
    {
        private readonly double _decay;
        private readonly double _rate;
        private readonly double _minRate;

        public LinearlyDecayingLearningRate(double initialRate, double decayRate, double minRate)
        {
            _rate = initialRate;
            _decay = decayRate;
            _minRate = minRate;
        }

        public T CalculateLearningRate(int layer, int epoch)
        {
            var rate = _rate*(Math.Pow(_decay, epoch));
            rate = Math.Max(rate, _minRate);
            return (T) Convert.ChangeType(rate, typeof (T));
        }
    }
}