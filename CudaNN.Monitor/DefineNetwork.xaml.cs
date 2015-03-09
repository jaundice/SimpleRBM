using System.Windows;
using CudaNN.DeepBelief.LayerBuilders;
using CudaNN.DeepBelief.ViewModels;
using Microsoft.Win32;

namespace CudaNN.DeepBelief
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

        private void LoadBinaryLayerFromFile(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { DefaultExt = ".dat", Filter = "Dat Files (.dat)|*.dat" };

            var res = dlg.ShowDialog(this);
            if (res == true)
            {
                ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Add(new LoadLayerInfo()
                {
                    Path = dlg.FileName,
                    LayerType = LoadLayerType.Binary
                });
            }
        }
        private void LoadLinearLayerFromFile(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { DefaultExt = ".dat", Filter = "Dat Files (.dat)|*.dat" };

            var res = dlg.ShowDialog(this);
            if (res == true)
            {
                ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Add(new LoadLayerInfo()
                {
                    Path = dlg.FileName,
                    LayerType = LoadLayerType.Linear
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

        private void Cancel(object sender, RoutedEventArgs e)
        {
            this.DialogResult = false;
            Close();
        }

        private void ClearLayers(object sender, RoutedEventArgs e)
        {
            ((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Clear();
        }

        private void Next(object sender, RoutedEventArgs e)
        {
            if (((LayerBuilderViewModel)this.DataContext).LayerConstructionInfo.Count > 0)
            {
                DialogResult = true;
                Close();
            }
        }
    }
}
