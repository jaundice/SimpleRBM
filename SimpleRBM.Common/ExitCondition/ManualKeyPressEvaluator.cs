using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleRBM.Common.ExitCondition
{
    public class ManualKeyPressEvaluator<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        private readonly int _layerDepth;
        private readonly int _maxEpochs;
        private readonly T _minError;
        private bool _exit;
        private bool _exitOnNextLowest;
        private T _lowestErrorSeen;
        private CancellationTokenSource src;
        private int _epochsSinceLastErrorImprovement = 0;

        public ManualKeyPressEvaluator(int layerDepth, int maxEpochs, T minError)
        {
            _maxEpochs = maxEpochs;
            _minError = minError;
            _layerDepth = layerDepth;
        }

        public bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime)
        {
            T tempLowest = _lowestErrorSeen;

            if (Comparer<T>.Default.Compare(lastError, _minError) < 0)
            {
                src.Cancel();
                _exit = true;
            }
            if (epochNumber == 0)
            {
                _lowestErrorSeen = lastError;
            }
            else if (epochNumber > _maxEpochs)
            {
                _exitOnNextLowest = true;
                if (epochNumber > _maxEpochs + 1000)
                {
                    _exit = true;
                    Console.WriteLine("Max epochs passed and no improvement in error within 1000 extra epochs. Aborting");
                }
                else
                {
                    if (epochNumber == _maxEpochs + 1 || epochNumber % 20 == 0)
                        Console.WriteLine("Max epochs passed. Exiting next time error drops below {0:F6}", _lowestErrorSeen);
                }
            }

            if (Comparer<T>.Default.Compare(lastError, _lowestErrorSeen) < 0)
            {
                _lowestErrorSeen = lastError;
                _epochsSinceLastErrorImprovement = 0;
                if (_exitOnNextLowest)
                {
                    src.Cancel();
                    _exit = true;
                }
            }
            else
            {
                _epochsSinceLastErrorImprovement++;
            }

            if (_exit || epochNumber % 20 == 0)
                Console.WriteLine("Epoch: {0}\tLayer: {1}\tError: {2:F6}\tElapsed: {3}\tdelta: {4:F6}\tepochs since improvement: {5}", epochNumber,
                    _layerDepth, lastError, elapsedTime,
                    (double)Convert.ChangeType(tempLowest, typeof(double)) -
                    (double)Convert.ChangeType(lastError, typeof(double)), _epochsSinceLastErrorImprovement);

            return _exit;
        }

        public void Start()
        {
            _exit = false;
            _exitOnNextLowest = false;

            if (src != null)
                src.Cancel();

            src = new CancellationTokenSource();

            Task.Factory.StartNew(() =>
            {
                ConsoleKeyInfo key = Console.ReadKey();
                if (key.KeyChar == 'l' || key.KeyChar == 'L')
                {
                    _exitOnNextLowest = true;
                    Console.WriteLine("Exiting next time epoch error < {0}", _lowestErrorSeen);
                }
                else
                {
                    _exit = true;
                }
                src.Cancel();
            }, src.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }


        public void Stop()
        {
            src.Cancel();
        }
    }
}