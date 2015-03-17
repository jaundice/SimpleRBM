using System.Windows;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;
#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;
#endif
namespace CudaNN.DeepBelief.LayerBuilders
{
    public class ConstructLinearHiddenLayer : ConstructNewLayer
    {
        public static readonly DependencyProperty TrainRandStDevProperty =
            DependencyProperty.Register("TrainRandStDev", typeof (TElement),
                typeof (ConstructLinearHiddenLayer), new PropertyMetadata(0.5));


        public TElement TrainRandStDev
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement) GetValue(TrainRandStDevProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(TrainRandStDevProperty, value)).Wait(); }
        }
    }
}