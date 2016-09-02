using System.Windows;
#if USEFLOAT
using TElement = System.Single;
#else
using TElement = System.Double;
#endif
namespace CudaNN.DeepBelief.LayerBuilders
{
    public class ConstructLinearHiddenLayer : ConstructNewLayer
    {
        public static readonly DependencyProperty TrainRandStDevProperty =
            DependencyProperty.Register("TrainRandStDev", typeof (TElement),
                typeof (ConstructLinearHiddenLayer), new PropertyMetadata((TElement)1.0/3));


        public TElement TrainRandStDev
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement) GetValue(TrainRandStDevProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(TrainRandStDevProperty, value)).Wait(); }
        }
    }
}