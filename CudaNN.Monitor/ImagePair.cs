using System.Windows;
using System.Windows.Media.Imaging;

namespace CudaNN.Monitor
{
    public class ImagePair : DependencyObject
    {
        public static readonly DependencyProperty Item1Property = DependencyProperty.Register("Item1", typeof(BitmapSource), typeof(ImagePair), new PropertyMetadata(default(BitmapSource)));
        public static readonly DependencyProperty Item2Property = DependencyProperty.Register("Item2", typeof(BitmapSource), typeof(ImagePair), new PropertyMetadata(default(BitmapSource)));

        public BitmapSource Item1
        {
            get { return Dispatcher.InvokeIfRequired(() => (BitmapSource)GetValue(Item1Property)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(Item1Property, value)).Wait(); }
        }

        public BitmapSource Item2
        {
            get { return Dispatcher.InvokeIfRequired(() => (BitmapSource)GetValue(Item2Property)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(Item2Property, value)).Wait(); }
        }
    }
}