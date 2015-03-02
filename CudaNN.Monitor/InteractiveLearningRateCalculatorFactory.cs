using System.Windows;
using SimpleRBM.Common;

namespace CudaNN.Monitor
{
    public class InteractiveLearningRateCalculatorFactory<T> : DependencyObject, ILearningRateCalculatorFactory<T>
    {
        public static readonly DependencyProperty LearningRateProperty = DependencyProperty.Register("LearningRate",
            typeof(T), typeof(InteractiveLearningRateCalculatorFactory<T>), new PropertyMetadata(default(T)));

        public static readonly DependencyProperty InitialLearningRateProperty = DependencyProperty.Register("InitialLearningRate",
            typeof(T), typeof(InteractiveLearningRateCalculatorFactory<T>), new PropertyMetadata(default(T)));


        private InteractiveLearningRateCalculator<T> _active;

        public T LearningRate
        {
            get { return Dispatcher.InvokeIfRequired(() => (T)GetValue(LearningRateProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LearningRateProperty, value)).Wait(); }
        }

        public T InitialLearningRate
        {
            get { return Dispatcher.InvokeIfRequired(() => (T)GetValue(InitialLearningRateProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(InitialLearningRateProperty, value)).Wait(); }
        }

        public ILearningRateCalculator<T> Create(int layer)
        {
            LearningRate = InitialLearningRate;
            return _active ?? (_active = new InteractiveLearningRateCalculator<T>()
            {
                LearningRate = InitialLearningRate
            });
        }

        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);

            if (e.Property == LearningRateProperty)
            {
                if (_active != null)
                {
                    _active.LearningRate = (T)e.NewValue;
                }
            }
        }

        public InteractiveLearningRateCalculatorFactory(T initialLearningRate)
        {
            InitialLearningRate = initialLearningRate;

        }
    }
}