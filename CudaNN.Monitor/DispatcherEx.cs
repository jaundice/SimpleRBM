using System;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace CudaNN.DeepBelief
{
    public static class DispatcherEx
    {
        public static Task InvokeIfRequired(this Dispatcher self, Action action)
        {
            if (self.CheckAccess())
            {
                action();
                return Task.Factory.StartNew(() => { });
            }
            else
            {
                return Task.Factory.StartNew(() => self.BeginInvoke(action).Wait());
                //self.InvokeAsync(action).Wait();
            }
        }

        public static Task<T> InvokeIfRequired<T>(this Dispatcher self, Func<T> action)
        {
            if (self.CheckAccess())
            {
                var res = action();
                return Task.Factory.StartNew(() => res);
            }
            return Task.Factory.StartNew<T>(() => (T)self.BeginInvoke(action).Result);
        }


    }
}