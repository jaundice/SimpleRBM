using System.Windows;

namespace CudaNN.DeepBelief.LayerBuilders
{
    public class ConstructLinearHiddenLayer : ConstructNewLayer
    {
        public static readonly DependencyProperty TrainRandStDevProperty =
            DependencyProperty.Register("TrainRandStDev", typeof (double),
                typeof (ConstructLinearHiddenLayer), new PropertyMetadata(0.5));


        public double TrainRandStDev
        {
            get { return Dispatcher.InvokeIfRequired(() => (double) GetValue(TrainRandStDevProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(TrainRandStDevProperty, value)).Wait(); }
        }
    }
}