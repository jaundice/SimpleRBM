using System.Windows;
using System.Windows.Controls;
using CudaNN.DeepBelief.ViewModels;

namespace CudaNN.DeepBelief.TemplateSelectors
{
    public class DataFieldTemplateSelector : DataTemplateSelector
    {
        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            var field = (FieldDefinitionViewModel)item;
            var res = Window.GetWindow(container).Resources;
            if (!field.IsEnabled)
                return res["DisabledFieldTemplate"] as DataTemplate;
            if (field.FieldType == FieldTypes.RealValue)
                return res["RealValueFieldTemplate"] as DataTemplate;
            if (field.FieldType == FieldTypes.OneOfNOptions)
                return res["OneOfNFieldTemplate"] as DataTemplate;

            return base.SelectTemplate(item, container);
        }
    }
}
