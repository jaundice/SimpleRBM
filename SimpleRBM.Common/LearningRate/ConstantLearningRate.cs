using System;

namespace SimpleRBM.Common.LearningRate
{
    public class ConstantLearningRate<T> : LearningRateCalculatorBase<T>
    {
        public ConstantLearningRate(double rate, int notify)
            : base(rate, notify)
        {

        }

        public override T DoCalculateLearningRate(int layer, int epoch)
        {
            return (T)Convert.ChangeType(_rate, typeof(T));
        }

    }
}