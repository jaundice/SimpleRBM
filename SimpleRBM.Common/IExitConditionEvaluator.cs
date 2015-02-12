using System;

namespace SimpleRBM.Common
{
    public interface IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime);
        void Start();
        void Stop();
    }
}