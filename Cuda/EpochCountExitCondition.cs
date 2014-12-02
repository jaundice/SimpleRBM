﻿using System;

namespace CudaRbm
{
    public class EpochCountExitCondition<T> : IExitConditionEvaluator<T> where T : IComparable<T>
    {
        public int MaxEpoch { get; set; }
        public int CurrentEpoch { get; protected set; }

        public bool Exit(int epochNumber, T lastError)
        {
            CurrentEpoch++;
            return CurrentEpoch > MaxEpoch;
        }

        public void Reset()
        {
            CurrentEpoch = 0;
        }
    }
}