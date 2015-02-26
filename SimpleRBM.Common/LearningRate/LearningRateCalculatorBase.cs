using System;

namespace SimpleRBM.Common.LearningRate
{
    public abstract class LearningRateCalculatorBase<T> : ILearningRateCalculator<T>
    {
        protected double _rate;
        private int _notifyFreq;

        public abstract T DoCalculateLearningRate(int layer, int epoch);

        protected LearningRateCalculatorBase(double rate, int notifyFreq = -1)
        {
            _rate = rate;
            _notifyFreq = notifyFreq;
        }

        public T CalculateLearningRate(int layer, int epoch)
        {
            T learningRate = DoCalculateLearningRate(layer, epoch);

            if (_notifyFreq > 0 && epoch % _notifyFreq == 0)
            {
                Console.WriteLine("Learning Rate: {0}", learningRate);
            }

            return learningRate;
        }
    }
}