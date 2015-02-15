using System;
using System.Collections;
using System.Collections.Generic;

namespace SimpleRBM.Common.ExitCondition
{
    public class EpochCountExitCondition<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        private T _lowestErrorSeen ;
        private readonly IEpochErrorTracker<T> _epochErrorTracker;
        public int MaxEpoch { get; set; }
        public int CurrentEpoch { get; protected set; }
        public int LayerDepth { get; set; }

        public EpochCountExitCondition(IEpochErrorTracker<T> epochErrorTracker)
        {
            _epochErrorTracker = epochErrorTracker;
        } 

        public bool Exit( int epochNumber, T lastError, TimeSpan elapsedTime)
        {
            _epochErrorTracker.LogEpochError(LayerDepth, epochNumber, lastError);

            if (epochNumber % 20 == 0)
                Console.WriteLine("Epoch: {0}\tLayer: {1}\tError: {2}\tElapsed: {3}, delta: {4}", epochNumber,
                    LayerDepth, lastError, elapsedTime,
                    (double)Convert.ChangeType(_lowestErrorSeen, typeof(double)) -
                    (double)Convert.ChangeType(lastError, typeof(double))); 
            
            CurrentEpoch++;

            if (epochNumber == 0)
            {
                _lowestErrorSeen = lastError;
            }
            else
            {
                if (Comparer<T>.Default.Compare(lastError, _lowestErrorSeen) < 0)
                {
                    _lowestErrorSeen = lastError;
                }
            }

            return CurrentEpoch > MaxEpoch;
        }

        public void Start()
        {
            CurrentEpoch = 0;
        }


        public void Stop()
        {
            
        }
    }
}