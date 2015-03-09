using System;
using System.Collections.Generic;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;

namespace CudaNN.DeepBelief
{

    public class InteractiveExitEvaluator<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        internal int _layerIndex;
        internal int _maxEpochs;
        private readonly T _minError;
        internal bool _exit;
        internal bool _exitOnNextLowest;
        private T _lowestErrorSeen;
        private int _epochsSinceLastErrorImprovement = 0;
        private readonly IEpochErrorTracker<T> _epochErrorTracker;

        public InteractiveExitEvaluator(IEpochErrorTracker<T> epochErrorTracker, int layerIndex, int maxEpochs,
            T minError)
        {
            _maxEpochs = maxEpochs;
            _minError = minError;
            _layerIndex = layerIndex;
            _epochErrorTracker = epochErrorTracker;
        }

        public bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime, out T delta)
        {



            T tempLowest = _lowestErrorSeen;

            delta = (T)Convert.ChangeType((double)Convert.ChangeType(tempLowest, typeof(double)) -
                                           (double)Convert.ChangeType(lastError, typeof(double)), typeof(T));

            _epochErrorTracker.LogEpochError(_layerIndex, epochNumber, lastError, delta, elapsedTime);

            if (Comparer<T>.Default.Compare(lastError, _minError) < 0)
            {
                _exit = true;
            }
            if (epochNumber == 0)
            {
                _lowestErrorSeen = lastError;
            }
            else if (epochNumber > _maxEpochs)
            {
                _exitOnNextLowest = true;
                if (epochNumber > _maxEpochs + 1000)
                {
                    _exit = true;
                    Console.WriteLine("Max epochs passed and no improvement in error within 1000 extra epochs. Aborting");
                }
            }

            if (Comparer<T>.Default.Compare(lastError, _lowestErrorSeen) < 0)
            {
                _lowestErrorSeen = lastError;
                _epochsSinceLastErrorImprovement = 0;
                if (_exitOnNextLowest)
                {
                    _exit = true;
                }
            }
            else
            {
                _epochsSinceLastErrorImprovement++;
            }


            return _exit;
        }

        public void Start()
        {
            _exit = false;
            _exitOnNextLowest = false;
        }




        public void Stop()
        {
        }
    }
}