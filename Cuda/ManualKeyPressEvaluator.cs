using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Mono.CSharp;

namespace CudaRbm
{
    public class ManualKeyPressEvaluator<T> : IExitConditionEvaluator<T> where T : IComparable<T>
    {
        private bool _exit;
        private readonly CancellationTokenSource src = new CancellationTokenSource();
        private readonly int _maxEpochs;
        private readonly T _minError;

        public bool Exit(int epochNumber, T lastError)
        {
            if (epochNumber > _maxEpochs || Comparer<T>.Default.Compare(lastError, _minError) < 0)
            {
                src.Cancel();
                _exit = true;
            }

            return _exit;
        }

        public ManualKeyPressEvaluator(int maxEpochs, T minError)
        {
            _maxEpochs = maxEpochs;
            _minError = minError;
        }

        public void Reset()
        {
            Task.Factory.StartNew(() =>
            {
                Console.ReadKey();
                _exit = true;
                src.Cancel();
            }, src.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }
    }
}