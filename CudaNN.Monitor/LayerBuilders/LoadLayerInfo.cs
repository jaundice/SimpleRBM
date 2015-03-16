using System.Windows;
using System.Windows.Input;

namespace CudaNN.DeepBelief.LayerBuilders
{
    public enum LoadLayerType
    {
        Binary,
        Linear
    }

    public class LoadLayerInfo : ConstructLayerBase
    {
        public static readonly DependencyProperty PathProperty =
            DependencyProperty.Register("Path", typeof (string),
                typeof (ConstructNewLayer), new PropertyMetadata(default(string)));

        public static readonly DependencyProperty LoadLayerTypeProperty =
            DependencyProperty.Register("LayerType", typeof (LoadLayerType),
                typeof (ConstructNewLayer), new PropertyMetadata(LoadLayerType.Binary));



        public string Path
        {
            get { return Dispatcher.InvokeIfRequired(() => (string) GetValue(PathProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(PathProperty, value)).Wait(); }
        }

        public LoadLayerType LayerType
        {
            get { return Dispatcher.InvokeIfRequired(() => (LoadLayerType) GetValue(LoadLayerTypeProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LoadLayerTypeProperty, value)).Wait(); }
        }

    }
}