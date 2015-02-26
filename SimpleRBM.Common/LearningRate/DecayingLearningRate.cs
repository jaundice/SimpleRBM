using System;
using System.Diagnostics;

namespace SimpleRBM.Common.LearningRate
{
    public class DecayingLearningRate<T> : LearningRateCalculatorBase<T>
    {
        private readonly double _decay;
        private readonly double _minRate;

        public delegate double Decay(double decayRate, int epoch);

        private readonly Decay _decayFunc;

        public DecayingLearningRate(double initialRate, double decayRate, double minRate, DecayType decayType, int notifyFreq)
            : this(initialRate, decayRate, minRate, GetDecay(decayType), notifyFreq)
        {

        }

        public DecayingLearningRate(double initialRate, double decayRate, double minRate, int notifyFreq)
            : this(initialRate, decayRate, minRate, DecayType.Power, notifyFreq)
        {

        }

        public DecayingLearningRate(double initialRate, double decayRate, double minRate, Decay decayFunc, int notifyFreq)
            : base(initialRate, notifyFreq)
        {
            _rate = initialRate;
            _decay = decayRate;
            _minRate = minRate;
            _decayFunc = decayFunc;
        }

        public override T DoCalculateLearningRate(int layer, int epoch)
        {
            var rate = _rate * _decayFunc(_decay, epoch);
            rate = Math.Max(rate, _minRate);
            return (T)Convert.ChangeType(rate, typeof(T));
        }

        public static Decay GetDecay(DecayType type)
        {
            switch (type)
            {
                case DecayType.Power:
                    return (a, b) => Math.Pow(a, b);
                case DecayType.HalfPower:
                    return (a, b) => Math.Pow(a, b < 2 ? 1 : b / 2.0);
                case DecayType.DoublePower:
                    return (a, b) => Math.Pow(a, b * 2);
                default:
                    throw new NotImplementedException();
            }
        }
    }
}