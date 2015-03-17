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
    public class ConstructBinaryLayer : ConstructNewLayer
    {
        public static readonly DependencyProperty EncodingNoiseLevelProperty =
            DependencyProperty.Register("EncodingNoiseLevel", typeof (TElement),
                typeof (ConstructBinaryLayer), new PropertyMetadata((TElement)1.0));


        public static readonly DependencyProperty DecodingNoiseLevelProperty =
            DependencyProperty.Register("DecodingNoiseLevel", typeof (TElement),
                typeof (ConstructBinaryLayer), new PropertyMetadata((TElement)1.0));


        public static readonly DependencyProperty ConvertActivationsToStatesProperty =
            DependencyProperty.Register("ConvertActivationsToStates", typeof (bool),
                typeof (ConstructBinaryLayer), new PropertyMetadata(true));

        public bool ConvertActivationsToStates
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (bool) GetValue(ConvertActivationsToStatesProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ConvertActivationsToStatesProperty, value)).Wait(); }
        }

        public TElement EncodingNoiseLevel
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement) GetValue(EncodingNoiseLevelProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(EncodingNoiseLevelProperty, value)).Wait(); }
        }

        public TElement DecodingNoiseLevel
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement) GetValue(DecodingNoiseLevelProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DecodingNoiseLevelProperty, value)).Wait(); }
        }
    }
}