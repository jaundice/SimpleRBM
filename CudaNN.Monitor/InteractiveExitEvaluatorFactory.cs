using System;
using System.Windows;
using System.Windows.Input;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;

namespace CudaNN.DeepBelief
{
    public class InteractiveExitEvaluatorFactory<T> : DependencyObject, IExitConditionEvaluatorFactory<T> where T : struct, IComparable<T>
    {
        public static readonly DependencyProperty ExitNextCommandBindingProperty =
            DependencyProperty.Register("ExitNextCommand", typeof(ICommand), typeof(InteractiveExitEvaluatorFactory<T>),
                new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty ExitNowCommandBindingProperty =
            DependencyProperty.Register("ExitNowCommand", typeof(ICommand), typeof(InteractiveExitEvaluatorFactory<T>),
                new PropertyMetadata(default(ICommand)));


        public static readonly DependencyProperty MaxEpochsProperty = DependencyProperty.Register("MaxEpochs", typeof(int), typeof(InteractiveExitEvaluatorFactory<T>), new PropertyMetadata(5000));


        public int MaxEpochs
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(MaxEpochsProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(MaxEpochsProperty, value)).Wait(); }
        }

        private readonly T _minError;
        private readonly IEpochErrorTracker<T> _epochErrorTracker;
        private InteractiveExitEvaluator<T> _activeEvaluator;

        public InteractiveExitEvaluatorFactory(IEpochErrorTracker<T> epochErrorTracker, T minError,
            int maxEpochs = 100000)
        {
            MaxEpochs = maxEpochs;
            _minError = minError;
            _epochErrorTracker = epochErrorTracker;

            var m = new Func<InteractiveExitEvaluator<T>>(() => this._activeEvaluator);

            ExitNextCommand = new CommandHandler(a =>
                _activeEvaluator._exitOnNextLowest = true,
                a => true);
            ExitNowCommand = new CommandHandler(a =>
               _activeEvaluator._exit = true,
                a => true);
        }

        public ICommand ExitNextCommand
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (ICommand)GetValue(ExitNextCommandBindingProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ExitNextCommandBindingProperty, value)).Wait(); }
        }

        public ICommand ExitNowCommand
        {
            get { return Dispatcher.InvokeIfRequired(() => (ICommand)GetValue(ExitNowCommandBindingProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ExitNowCommandBindingProperty, value)).Wait(); }
        }

        public IExitConditionEvaluator<T> Create(int layerIndex)
        {
            _activeEvaluator = _activeEvaluator ?? (_activeEvaluator = new InteractiveExitEvaluator<T>(_epochErrorTracker, layerIndex, MaxEpochs,
                _minError));
            _activeEvaluator._layerIndex = layerIndex;

            return _activeEvaluator;
        }

        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);
            if (e.Property == MaxEpochsProperty)
            {
                if (_activeEvaluator != null)
                {
                    _activeEvaluator._maxEpochs = (int)e.NewValue;
                }
            }
        }
    }
}