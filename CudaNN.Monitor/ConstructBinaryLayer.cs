using System.Windows;

namespace CudaNN.Monitor
{
    public class ConstructBinaryLayer : ConstructNewLayer
    {
        public static readonly DependencyProperty EncodingNoiseLevelProperty =
            DependencyProperty.Register("EncodingNoiseLevel", typeof (double),
                typeof (ConstructBinaryLayer), new PropertyMetadata(1.0));


        public static readonly DependencyProperty DecodingNoiseLevelProperty =
            DependencyProperty.Register("DecodingNoiseLevel", typeof (double),
                typeof (ConstructBinaryLayer), new PropertyMetadata(1.0));


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

        public double EncodingNoiseLevel
        {
            get { return Dispatcher.InvokeIfRequired(() => (double) GetValue(EncodingNoiseLevelProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(EncodingNoiseLevelProperty, value)).Wait(); }
        }

        public double DecodingNoiseLevel
        {
            get { return Dispatcher.InvokeIfRequired(() => (double) GetValue(DecodingNoiseLevelProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DecodingNoiseLevelProperty, value)).Wait(); }
        }
    }
}