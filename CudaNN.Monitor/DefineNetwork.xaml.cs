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
using System.Windows.Shapes;
using Microsoft.Win32;

namespace CudaNN.Monitor
{
    /// <summary>
    /// Interaction logic for DefineNetwork.xaml
    /// </summary>
    public partial class DefineNetwork : Window
    {
        public DefineNetwork()
        {
            InitializeComponent();
        }

        private void LoadLayerFromFile(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { DefaultExt = ".dat", Filter = "Dat Files (.dat)|*.dat" };

            var res = dlg.ShowDialog(this);
            if (res == true)
            {
                ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Add(new LoadLayerInfo()
                {
                    Path = dlg.FileName
                });
            }
        }

        private void CreateBinaryLayer(object sender, RoutedEventArgs e)
        {
            ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Add(new ConstructBinaryLayer()
            {
                LayerIndex = ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Count
            });
        }

        private void CreateLinearLayer(object sender, RoutedEventArgs e)
        {
            ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Add(new ConstructLinearHiddenLayer()
            {
                LayerIndex = ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Count
            });
        }

        private void CloseWindow(object sender, RoutedEventArgs e)
        {
            this.DialogResult = true;
            Close();
        }

        private void ClearLayers(object sender, RoutedEventArgs e)
        {
            ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Clear();
        }
    }
}
