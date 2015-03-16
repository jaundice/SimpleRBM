using System;
using System.Collections.Generic;
using System.Diagnostics;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;

namespace CudaNN.DeepBelief
{

    public class InteractiveExitEvaluator<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        private T _lowestErrorSeen;
        private readonly IEpochErrorTracker<T> _epochErrorTracker;

        public InteractiveExitEvaluator(IEpochErrorTracker<T> epochErrorTracker, int layerIndex, int maxEpochs,
            T minError)
        {
            EpochsSinceLastErrorImprovement = 0;
            MaxEpochs = maxEpochs;
            MinError = minError;
            LayerIndex = layerIndex;
            _epochErrorTracker = epochErrorTracker;
        }

        public int LayerIndex { get; protected internal set; }

        public int MaxEpochs { get; protected internal set; }

        public T MinError { get; protected internal set; }

        public bool ExitNextEpoch { get; protected internal set; }

        public bool ExitOnNextLowestError { get; protected internal set; }

        public int EpochsSinceLastErrorImprovement { get; protected set; }

        public bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime, out T delta)
        {



            T tempLowest = _lowestErrorSeen;

            delta = (T)Convert.ChangeType((double)Convert.ChangeType(tempLowest, typeof(double)) -
                                           (double)Convert.ChangeType(lastError, typeof(double)), typeof(T));

            _epochErrorTracker.LogEpochError(LayerIndex, epochNumber, lastError, delta, elapsedTime);

            if (Comparer<T>.Default.Compare(lastError, MinError) < 0)
            {
                ExitNextEpoch = true;
            }
            if (epochNumber == 0)
            {
                _lowestErrorSeen = lastError;
            }
            else if (epochNumber > MaxEpochs)
            {
                ExitOnNextLowestError = true;
                if (epochNumber > MaxEpochs + 1000)
                {
                    ExitNextEpoch = true;
                    Trace.WriteLine("Max epochs passed and no improvement in error within 1000 extra epochs. Aborting");
                }
            }

            if (Comparer<T>.Default.Compare(lastError, _lowestErrorSeen) < 0)
            {
                _lowestErrorSeen = lastError;
                EpochsSinceLastErrorImprovement = 0;
                if (ExitOnNextLowestError)
                {
                    ExitNextEpoch = true;
                }
            }
            else
            {
                EpochsSinceLastErrorImprovement++;
            }


            return ExitNextEpoch;
        }

        public void Start()
        {
            ExitNextEpoch = false;
            ExitOnNextLowestError = false;
        }




        public void Stop()
        {
        }
    }
}