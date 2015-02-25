using System;
using System.Diagnostics;

namespace SimpleRBM.Common.LearningRate
{
    public class DecayingLearningRate<T> : ILearningRateCalculator<T>
    {
        private readonly double _decay;
        private readonly double _rate;
        private readonly double _minRate;

        public delegate double Decay(double decayRate, int epoch);

        private readonly Decay _decayFunc;

        public DecayingLearningRate(double initialRate, double decayRate, double minRate, DecayType decayType)
            : this(initialRate, decayRate, minRate, GetDecay(decayType))
        {

        }

        public DecayingLearningRate(double initialRate, double decayRate, double minRate)
            : this(initialRate, decayRate, minRate, DecayType.Power)
        {

        }

        public DecayingLearningRate(double initialRate, double decayRate, double minRate, Decay decayFunc)
        {
            _rate = initialRate;
            _decay = decayRate;
            _minRate = minRate;
            _decayFunc = decayFunc;
        }

        public T CalculateLearningRate(int layer, int epoch)
        {
            var rate = _rate * _decayFunc(_decay, epoch);
            rate = Math.Max(rate, _minRate);
#if DEBUG
            if (epoch % 20 == 0)
                Console.WriteLine("Learning Rate: {0}", rate);
#endif
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
                default:
                    throw new NotImplementedException();
            }
        }
    }
}