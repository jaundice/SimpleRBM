using System.Windows;
using System.Windows.Controls;
using CudaNN.DeepBelief.ViewModels;

namespace CudaNN.DeepBelief
{
    /// <summary>
    ///     Interaction logic for ConfigureData.xaml
    /// </summary>
    public partial class ConfigureData : Window
    {
        public ConfigureData()
        {
            InitializeComponent();
        }


        private void Selector_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var lb = sender as ComboBox;
            if ((string)((ComboBoxItem)lb.SelectedItem).Content == "Kaggle")
                DataContext = new ImageDataConfigViewModel();
            else if ((string)((ComboBoxItem)lb.SelectedItem).Content == "Images")
                DataContext = new ImageDataConfigViewModel();
            else if ((string)((ComboBoxItem)lb.SelectedItem).Content == "Text")
                DataContext = new TextDataConfigViewModel();
        }

        private void Submit(object sender, RoutedEventArgs e)
        {
            if (!((DataConfigViewModelBase)DataContext).Validate())
                return;
            DialogResult = true;
            Close();
        }

        private void Cancel(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}