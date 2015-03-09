using System.Windows;

namespace CudaNN.DeepBelief
{
    /// <summary>
    /// Interaction logic for ConfigureLearningRates.xaml
    /// </summary>
    public partial class ConfigureLearningRates : Window
    {
        public ConfigureLearningRates()
        {
            InitializeComponent();
        }

        private void Next(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
