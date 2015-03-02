using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleRBM.Common.ExitCondition
{
    public class ConsoleKeyListener
    {
        public event EventHandler<KeyPressEventArgs> KeyPress;

        private static ConsoleKeyListener _instance = new ConsoleKeyListener();

        public static ConsoleKeyListener Instance
        {
            get { return _instance ?? (_instance = new ConsoleKeyListener()); }
        }

        private Task _task;

        private ConsoleKeyListener()
        {
            _task = Task.Run(() =>
            {
                while (true)
                {
                    OnKeyPress(Console.ReadKey(true));
                }
            });
        }

        private void OnKeyPress(ConsoleKeyInfo consoleKeyInfo)
        {
            if (KeyPress != null)
            {
                KeyPress(this, new KeyPressEventArgs(consoleKeyInfo));
            }
        }
    }

    public class ManualKeyPressEvaluator<T> : IExitConditionEvaluator<T> where T : struct, IComparable<T>
    {
        private readonly int _layerDepth;
        private readonly int _maxEpochs;
        private readonly T _minError;
        private bool _exit;
        private bool _exitOnNextLowest;
        private T _lowestErrorSeen;
        private int _epochsSinceLastErrorImprovement = 0;
        private IEpochErrorTracker<T> _epochErrorTracker;
        private int _reportFrequency;

        public ManualKeyPressEvaluator(IEpochErrorTracker<T> epochErrorTracker, int layerDepth, int maxEpochs,
            T minError, int reportFrequency)
        {
            _maxEpochs = maxEpochs;
            _minError = minError;
            _layerDepth = layerDepth;
            _epochErrorTracker = epochErrorTracker;
            _reportFrequency = reportFrequency;
        }

        public bool Exit(int epochNumber, T lastError, TimeSpan elapsedTime, out T delta)
        {
            


            T tempLowest = _lowestErrorSeen;

            delta = (T) Convert.ChangeType((double) Convert.ChangeType(tempLowest, typeof (double)) -
                                           (double) Convert.ChangeType(lastError, typeof (double)), typeof (T));

            _epochErrorTracker.LogEpochError(_layerDepth, epochNumber, lastError, delta, elapsedTime);

            if (Comparer<T>.Default.Compare(lastError, _minError) < 0)
            {
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
                    if (epochNumber == _maxEpochs + 1 || epochNumber % _reportFrequency == 0)
                        Console.WriteLine("Max epochs passed. Exiting next time error drops below {0:F6}",
                            _lowestErrorSeen);
                }
            }

            if (Comparer<T>.Default.Compare(lastError, _lowestErrorSeen) < 0)
            {
                _lowestErrorSeen = lastError;
                _epochsSinceLastErrorImprovement = 0;
                if (_exitOnNextLowest)
                {
                    _exit = true;
                }
            }
            else
            {
                _epochsSinceLastErrorImprovement++;
            }

            if (_exit || epochNumber % _reportFrequency == 0)
                Console.WriteLine(
                    "Epoch: {0}\tLayer: {1}\tError: {2:F6}\tElapsed: {3}\tdelta: {4:F6}\tepochs since improvement: {5}",
                    epochNumber,
                    _layerDepth, lastError, elapsedTime,delta, _epochsSinceLastErrorImprovement);



            return _exit;
        }

        public void Start()
        {
            _exit = false;
            _exitOnNextLowest = false;
            ConsoleKeyListener.Instance.KeyPress += Instance_KeyPress;
        }

        private void Instance_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyInfo.KeyChar == 'l' || e.KeyInfo.KeyChar == 'L')
            {
                _exitOnNextLowest = true;
                Console.WriteLine("Exiting next time epoch error < {0}", _lowestErrorSeen);
            }
            else
            {
                Console.WriteLine("Exiting now");

                _exit = true;
            }
        }


        public void Stop()
        {
            ConsoleKeyListener.Instance.KeyPress -= Instance_KeyPress;
        }
    }
}