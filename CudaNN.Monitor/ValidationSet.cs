using System.Windows;
using System.Windows.Media.Imaging;

namespace CudaNN.DeepBelief
{

    public class ImageSet : DependencyObject
    {
        public static readonly DependencyProperty DataImageProperty = DependencyProperty.Register("DataImage",
            typeof(BitmapSource), typeof(ImageSet), new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty CodeImageProperty =
            DependencyProperty.Register("CodeImage",
                typeof(BitmapSource), typeof(ImageSet), new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty LabelProperty =
            DependencyProperty.Register("Label",
                typeof(string), typeof(ImageSet), new PropertyMetadata(default(string)));


        public BitmapSource DataImage
        {
            get { return Dispatcher.InvokeIfRequired(() => (BitmapSource)GetValue(DataImageProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DataImageProperty, value)).Wait(); }
        }

        public BitmapSource CodeImage
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (BitmapSource)GetValue(CodeImageProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(CodeImageProperty, value)).Wait(); }
        }

        public string Label
        {
            get { return Dispatcher.InvokeIfRequired(() => (string)GetValue(DataImageProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LabelProperty, value)).Wait(); }
        }


    }


    public class ValidationSet : DependencyObject
    {
        public static readonly DependencyProperty OriginalImageSetProperty = DependencyProperty.Register("OriginalImageSet",
            typeof(ImageSet), typeof(ValidationSet), new PropertyMetadata(default(ImageSet)));

        public static readonly DependencyProperty ReconstructedImageSetProperty = DependencyProperty.Register("ReconstructedImageSet",
           typeof(ImageSet), typeof(ValidationSet), new PropertyMetadata(default(ImageSet)));

        public ImageSet OriginalImageSet
        {
            get { return Dispatcher.InvokeIfRequired(() => (ImageSet)GetValue(OriginalImageSetProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(OriginalImageSetProperty, value)).Wait(); }
        }

        public ImageSet ReconstructedImageSet
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (ImageSet)GetValue(ReconstructedImageSetProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ReconstructedImageSetProperty, value)).Wait(); }
        }
    }
}