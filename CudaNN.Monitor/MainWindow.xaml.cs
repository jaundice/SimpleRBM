using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace CudaNN.Monitor
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void FrameworkElement_OnSourceUpdated(object sender, DataTransferEventArgs e)
        {
            Image im = e.Source as Image;
            if (im != null)
            {
                RenderOptions.SetBitmapScalingMode(im, BitmapScalingMode.NearestNeighbor);
            }
        }

        private void FrameworkElement_OnLoaded(object sender, RoutedEventArgs e)
        {
            Image im = e.Source as Image;
            if (im != null)
            {
                RenderOptions.SetBitmapScalingMode(im, BitmapScalingMode.NearestNeighbor);
            }
        }
    }
}
