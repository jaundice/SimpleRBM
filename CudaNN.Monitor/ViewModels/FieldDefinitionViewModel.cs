using System;
using System.Windows;

namespace CudaNN.DeepBelief.ViewModels
{
    public class FieldDefinitionViewModel : DependencyObject
    {
        public event EventHandler FieldChanged;


        public static readonly DependencyProperty IsEnabledProperty =
            DependencyProperty.Register("IsEnabled", typeof(bool),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(bool)));

        public static readonly DependencyProperty IsLabelsProperty =
            DependencyProperty.Register("IsLabels", typeof(bool),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(bool)));

        public static readonly DependencyProperty FieldTypeProperty =
            DependencyProperty.Register("FieldType", typeof(FieldTypes),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(FieldTypes.RealValue));

        public static readonly DependencyProperty ParseErrorsProperty =
            DependencyProperty.Register("ParseErrors", typeof(int),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(int)));


        public static readonly DependencyProperty FieldWidthProperty =
            DependencyProperty.Register("FieldWidth", typeof(int),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty SourceIndexProperty =
            DependencyProperty.Register("SourceIndex", typeof(int),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty FieldNameProperty =
            DependencyProperty.Register("FieldName", typeof(string),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(string)));

        public static readonly DependencyProperty MinRealValueProperty =
            DependencyProperty.Register("MinRealValue", typeof(double),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(double)));

        public static readonly DependencyProperty MaxRealValueProperty =
            DependencyProperty.Register("MaxRealValue", typeof(double),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(double)));

        public static readonly DependencyProperty OneOfNOptionsProperty =
            DependencyProperty.Register("OneOfNOptions", typeof(string[]),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(string[])));

        public static readonly DependencyProperty UseGrayCodeForOneOfNOptionsProperty =
            DependencyProperty.Register("UseGrayCodeForOneOfNOptions", typeof(bool),
                typeof(FieldDefinitionViewModel), new PropertyMetadata(default(bool)));


        private static readonly FieldTypes[] _fieldTypes = new[] { FieldTypes.RealValue, FieldTypes.OneOfNOptions };
        public static FieldTypes[] AllFieldTypes
        {
            get { return _fieldTypes; }
        }

        public FieldDefinitionViewModel()
        {
            FieldType = FieldTypes.RealValue;
        }

        public string FieldName
        {
            get { return (string)GetValue(FieldNameProperty); }
            set { SetValue(FieldNameProperty, value); }
        }

        public bool IsEnabled
        {
            get { return (bool)GetValue(IsEnabledProperty); }
            set { SetValue(IsEnabledProperty, value); }
        }

        public bool IsLabels
        {
            get { return (bool)GetValue(IsLabelsProperty); }
            set { SetValue(IsLabelsProperty, value); }
        }

        public bool UseGrayCodeForOneOfNOptions
        {
            get { return (bool)GetValue(UseGrayCodeForOneOfNOptionsProperty); }
            set { SetValue(UseGrayCodeForOneOfNOptionsProperty, value); }
        }

        public FieldTypes FieldType
        {
            get { return (FieldTypes)GetValue(FieldTypeProperty); }
            set { SetValue(FieldTypeProperty, value); }
        }

        public int FieldWidth
        {
            get { return (int)GetValue(FieldWidthProperty); }
            set { SetValue(FieldWidthProperty, value); }
        }

        public int SourceIndex
        {
            get { return (int)GetValue(SourceIndexProperty); }
            set { SetValue(SourceIndexProperty, value); }
        }

        public double MinRealValue
        {
            get { return (double)GetValue(MinRealValueProperty); }
            set { SetValue(MinRealValueProperty, value); }
        }

        public double MaxRealValue
        {
            get { return (double)GetValue(MaxRealValueProperty); }
            set { SetValue(MaxRealValueProperty, value); }
        }

        public string[] OneOfNOptions
        {
            get { return (string[])GetValue(OneOfNOptionsProperty); }
            set { SetValue(OneOfNOptionsProperty, value); }
        }

        public int ParseErrors
        {
            get { return (int)GetValue(ParseErrorsProperty); }
            set { SetValue(ParseErrorsProperty, value); }
        }

        protected void OnFieldChanged()
        {
            if (this.FieldChanged != null)
                FieldChanged(this, EventArgs.Empty);
        }

        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);
            if (e.Property == FieldTypeProperty
                || e.Property == IsEnabledProperty
                || e.Property == IsLabelsProperty
                || e.Property == UseGrayCodeForOneOfNOptionsProperty)
                OnFieldChanged();
        }
    }
}