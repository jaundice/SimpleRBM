using System;

namespace SimpleRBM.Common.ExitCondition
{
    public interface IEpochErrorTracker<T>
    {
        void LogEpochError(int layer, int epoch, T error, T delta, TimeSpan elapsed);
    }
}