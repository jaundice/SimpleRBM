using System.Windows;
using System.Windows.Controls;
using CudaNN.DeepBelief.LayerBuilders;

namespace CudaNN.DeepBelief.TemplateSelectors
{
    public class LayerBuilderTemplateSelector : DataTemplateSelector
    {
        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {

            var window = Window.GetWindow(container);

            if (item is LoadLayerInfo)
            {
                return window.Resources["LoadLayerTemplate"] as DataTemplate;
            }
            if (item is ConstructBinaryLayer)
            {
                return window.Resources["ConstructBinaryLayerTemplate"] as DataTemplate;
            }
            if (item is ConstructLinearHiddenLayer)
            {
                return window.Resources["ConstructLinearHiddenLayerTemplate"] as DataTemplate;
            }

            return base.SelectTemplate(item, container);
        }
    }
}