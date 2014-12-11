﻿using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleRBM.Common.ExitCondition
{
    public class ManualKeyPressEvaluator<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        private bool _exit;
        private readonly CancellationTokenSource src = new CancellationTokenSource();
        private readonly int _maxEpochs;
        private readonly T _minError;
        private T _lowestErrorSeen;
        private bool _exitOnNextLowest;

        public bool Exit(int epochNumber, T lastError)
        {
            if (epochNumber > _maxEpochs || Comparer<T>.Default.Compare(lastError, _minError) < 0)
            {
                src.Cancel();
                _exit = true;
            }
            if (epochNumber == 0)
            {
                _lowestErrorSeen = lastError;

            }
            else if (Comparer<T>.Default.Compare(lastError, _lowestErrorSeen) < 0)
            {
                _lowestErrorSeen = lastError;
                if (_exitOnNextLowest)
                {
                    src.Cancel();
                    _exit = true;
                }
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
            _exit = false;
            _exitOnNextLowest = false;


            Task.Factory.StartNew(() =>
            {
                var key = Console.ReadKey();
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
    }
}