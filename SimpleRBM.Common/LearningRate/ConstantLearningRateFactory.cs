using System;

namespace SimpleRBM.Common.LearningRate
{
    public class ConstantLearningRateFactory<T> : ILearningRateCalculatorFactory<T>
    {
        private readonly T _rate;

        public ConstantLearningRateFactory(double rate)
        {
            _rate = (T) Convert.ChangeType(rate, typeof (T));
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return new ConstantLearningRate<T>(_rate);
        }
    }
}