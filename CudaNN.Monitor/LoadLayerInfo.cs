using System.Windows;

namespace CudaNN.Monitor
{
    public class LoadLayerInfo : ConstructLayerBase
    {
        public static readonly DependencyProperty PathProperty =
            DependencyProperty.Register("Path", typeof (string),
                typeof (ConstructNewLayer), new PropertyMetadata(default(string)));

        public string Path
        {
            get { return Dispatcher.InvokeIfRequired(() => (string) GetValue(PathProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(PathProperty, value)).Wait(); }
        }
    }
}