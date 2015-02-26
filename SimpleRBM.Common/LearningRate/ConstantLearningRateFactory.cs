using System;

namespace SimpleRBM.Common.LearningRate
{
    public class ConstantLearningRateFactory<T> : ILearningRateCalculatorFactory<T>
    {
        private readonly double _rate;
        private readonly int _notifyFreq;

        public ConstantLearningRateFactory(double rate, int notifyFreq=-1)
        {
            _rate = rate;
            _notifyFreq = notifyFreq;
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            return new ConstantLearningRate<T>(_rate, _notifyFreq);
        }
    }
}