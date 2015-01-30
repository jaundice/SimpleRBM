using System;

namespace SimpleRBM.Common.ExitCondition
{
    public class EpochCountExitCondition<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        public int MaxEpoch { get; set; }
        public int CurrentEpoch { get; protected set; }
        public int LayerDepth { get; set; }

        public bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime)
        {
            if (epochNumber % 20 == 0)
                Console.WriteLine("Epoch: {0}\tLayer: {1}\tError: {2}\tElapsed: {3}", epochNumber, LayerDepth, lastError, elapsedTime);
            CurrentEpoch++;
            return CurrentEpoch > MaxEpoch;
        }

        public void Reset()
        {
            CurrentEpoch = 0;
        }
    }
}