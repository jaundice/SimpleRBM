using System.Windows;
using System.Windows.Controls;

namespace CudaNN.Monitor
{
    public class LayerConfigViewerTemplateSelector : DataTemplateSelector
    {
        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {

            var window = Window.GetWindow(container);

            if (item is LoadLayerInfo)
            {
                return window.Resources["LoadedLayerTemplate"] as DataTemplate;
            }
            if (item is ConstructBinaryLayer)
            {
                return window.Resources["BinaryLayerTemplate"] as DataTemplate;
            }
            if (item is ConstructLinearHiddenLayer)
            {
                return window.Resources["LinearHiddenLayerTemplate"] as DataTemplate;
            }

            return base.SelectTemplate(item, container);
        }
    }
}