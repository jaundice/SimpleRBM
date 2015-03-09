using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace CudaNN.DeepBelief
{
    /// <summary>
    ///     Interaction logic for Believe.xaml
    /// </summary>
    public partial class Believe : Window
    {
        public Believe()
        {
            InitializeComponent();
        }

        private void FrameworkElement_OnLoaded(object sender, RoutedEventArgs e)
        {
            var im = e.Source as Image;
            if (im != null)
            {
                RenderOptions.SetBitmapScalingMode(im, BitmapScalingMode.NearestNeighbor);
            }
        }

        private void Image_OnImageFailed(object sender, ExceptionRoutedEventArgs e)
        {
            var im = e.Source as Image;
            Trace.TraceWarning("Error loading image");
        }
    }
}