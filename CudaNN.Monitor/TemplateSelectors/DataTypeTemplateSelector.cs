using System.Windows;
using System.Windows.Controls;
using CudaNN.DeepBelief.Properties;
using CudaNN.DeepBelief.ViewModels;

namespace CudaNN.DeepBelief.TemplateSelectors
{
    public class DataTypeTemplateSelector : DataTemplateSelector
    {
        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            var config = (DataConfigViewModelBase)item;
            var window = Window.GetWindow(container);
            var res = window != null ? window.Resources : new ConfigureData().Resources;//designer
            if (config is ImageDataConfigViewModel)
                return res["ImageSourceTemplate"] as DataTemplate;
            if (config is TextDataConfigViewModel)
                return res["FileSourceTemplate"] as DataTemplate;
            return base.SelectTemplate(item, container);
        }
    }
}