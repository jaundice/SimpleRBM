using System;

namespace SimpleRBM.Common.LearningRate
{
    public class LinearlyDecayingLearningRate<T> : ILearningRateCalculator<T>
    {
        private readonly double _decay;
        private readonly double _rate;

        public LinearlyDecayingLearningRate(double initialRate, double decayRate)
        {
            _rate = initialRate;
            _decay = decayRate;
        }

        public T CalculateLearningRate(int layer, int epoch)
        {
            return (T) Convert.ChangeType(_rate*(Math.Pow(_decay, epoch)), typeof (T));
        }
    }
}