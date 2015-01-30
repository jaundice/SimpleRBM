using System;

namespace SimpleRBM.Common.LearningRate
{
    public class ConstantLearningRate<T> : ILearningRateCalculator<T>
    {
        private readonly T _rate;

        public ConstantLearningRate(double rate)
        {
            _rate = (T)Convert.ChangeType(rate, typeof(T));
        }

        public T CalculateLearningRate(int layer, int epoch)
        {
            return _rate ;
        }
    }
}