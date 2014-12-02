using System;

namespace CudaRbm
{
    public interface IExitConditionEvaluator<T> where T : IComparable<T>
    {
        bool Exit(int epochNumber, T lastError);
        void Reset();
    }
}