using System;

namespace SimpleRBM.Common
{
    public interface IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime, out T delta);
        void Start();
        void Stop();
    }
}