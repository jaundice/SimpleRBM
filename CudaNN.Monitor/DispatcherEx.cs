using System;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace CudaNN.DeepBelief
{
    public static class DispatcherEx
    {
        public static async Task InvokeIfRequired(this Dispatcher self, Action action)
        {
            if (self.CheckAccess())
                action();
            else
                await self.InvokeAsync(action);
        }

        public static async Task<T> InvokeIfRequired<T>(this Dispatcher self, Func<T> action)
        {
            if (self.CheckAccess())
                return action();
            return await self.InvokeAsync(action);
        }

       
    }
}