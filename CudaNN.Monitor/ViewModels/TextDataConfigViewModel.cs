using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using CudaNN.DeepBelief.DataIO;
using Mono.CSharp;
using SimpleRBM.Demo.Util;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;
#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;
#endif
namespace CudaNN.DeepBelief.ViewModels
{
    public class TextDataConfigViewModel : DataConfigViewModelBase
    {
        //public static readonly DependencyProperty SelectedLineSeparatorProperty =
        //    DependencyProperty.Register("SelectedLineSeparator", typeof(System.Tuple<string, char[]>[]),
        //        typeof(TextDataConfigViewModel), new PropertyMetadata(default(System.Tuple<string, char[]>[])),
        //        a => a != null);

        public static readonly DependencyProperty SelectedFieldSeparatorProperty =
            DependencyProperty.Register("SelectedFieldSeparator", typeof (System.Tuple<string, char>),
                typeof (TextDataConfigViewModel), new PropertyMetadata(default(System.Tuple<string, char>)));

        public static readonly DependencyProperty FirstLineIsHeaderProperty =
            DependencyProperty.Register("FirstLineIsHeader", typeof (bool),
                typeof (TextDataConfigViewModel), new PropertyMetadata(default(bool)));

        public static readonly DependencyProperty FieldDefinitionsProperty =
            DependencyProperty.Register("FieldDefinitions", typeof (ObservableCollection<FieldDefinitionViewModel>),
                typeof (TextDataConfigViewModel),
                new PropertyMetadata(default(ObservableCollection<FieldDefinitionViewModel>)));


        public static readonly DependencyProperty ReparseFileCommandProperty =
            DependencyProperty.Register("ReparseFileCommand", typeof (ICommand),
                typeof (TextDataConfigViewModel),
                new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty UIIsOutOfSyncProperty =
            DependencyProperty.Register("UIIsOutOfSync", typeof (bool),
                typeof (TextDataConfigViewModel),
                new PropertyMetadata(default(bool)));

        public static readonly DependencyProperty LabelFieldExistsInTestSetProperty =
            DependencyProperty.Register("LabelFieldExistsInTestSet", typeof (bool),
                typeof (TextDataConfigViewModel),
                new PropertyMetadata(default(bool)));

        public DataTransformationTypes[] AllAvailableTransformationTypes
        {
            get
            {
                return new[]
                {
                    DataTransformationTypes.NoTransform, DataTransformationTypes.DivideByGlobalLargestFieldValue,
                    DataTransformationTypes.DivideByLargestFieldValue, DataTransformationTypes.DivideBy255,
                    DataTransformationTypes.Subtract128Divide127
                };
            }
        }


    public TextDataConfigViewModel()
        {
            SelectedFieldSeparator = FieldSeparators[0];
            FirstLineIsHeader = true;
            ReparseFileCommand = new CommandHandler(a => UpdateFieldInfo(), a => true);
        }

        public ICommand ReparseFileCommand
        {
            get { return (ICommand)GetValue(ReparseFileCommandProperty); }
            set { SetValue(ReparseFileCommandProperty, value); }
        }

        public bool LabelFieldExistsInTestSet
        {
            get { return (bool)GetValue(LabelFieldExistsInTestSetProperty); }
            set { SetValue(LabelFieldExistsInTestSetProperty, value); }
        }

        public bool UIIsOutOfSync
        {
            get { return (bool)GetValue(UIIsOutOfSyncProperty); }
            set { SetValue(UIIsOutOfSyncProperty, value); }
        }


        //public System.Tuple<string, char[]>[] LineSeparators
        //{
        //    get
        //    {
        //        return new[]
        //        {
        //            Tuple.Create("Carriage Return Line Feed", new[] {'\r', '\n'}),
        //            Tuple.Create("Carriage Return", new[] {'\r'}),
        //            Tuple.Create("Line Feeed", new[] {'\n'})
        //        };
        //    }
        //}

        public System.Tuple<string, char>[] FieldSeparators
        {
            get
            {
                return new[]
                {
                    Tuple.Create("Comma", ','),
                    Tuple.Create("Tab", '\t'),
                    Tuple.Create("Pipe", '|')
                };
            }
        }

        //public System.Tuple<string, char[]>[] SelectedLineSeparator
        //{
        //    get { return (System.Tuple<string, char[]>[])GetValue(SelectedLineSeparatorProperty); }
        //    set { SetValue(SelectedLineSeparatorProperty, value); }
        //}

        public System.Tuple<string, char> SelectedFieldSeparator
        {
            get { return (System.Tuple<string, char>)GetValue(SelectedFieldSeparatorProperty); }
            set { SetValue(SelectedFieldSeparatorProperty, value); }
        }

        public bool FirstLineIsHeader
        {
            get { return (bool)GetValue(FirstLineIsHeaderProperty); }
            set { SetValue(FirstLineIsHeaderProperty, value); }
        }


        public override DataContainerType ContainerType
        {
            get { return DataContainerType.File; }
        }

        public override string FileExtensionFilter
        {
            get { return "Comma Separated Values(.csv)|*.csv|Tab Separated Values (.tsv)|*.tsv|Text(.txt)|*.txt"; }
        }

        public ObservableCollection<FieldDefinitionViewModel> FieldDefinitions
        {
            get { return (ObservableCollection<FieldDefinitionViewModel>)GetValue(FieldDefinitionsProperty); }
            set { SetValue(FieldDefinitionsProperty, value); }
        }

        public override bool Validate()
        {
            if (!base.Validate())
                return false;

            ObservableCollection<FieldDefinitionViewModel> fields = FieldDefinitions;

            if (fields.Count(b => b.IsEnabled) == 0)
                return false;

            List<FieldDefinitionViewModel> labels = fields.Where(b => b.IsLabels).ToList();

            if (labels.Count > 1)
                return false;

            FieldDefinitionViewModel label = labels.FirstOrDefault();
            if (label != null && label.FieldType == FieldTypes.RealValue)
                return false;

            if (fields.Sum(a => a.ParseErrors) > 0)
                return false;

            if (TrainingRecordCount == 0 || TestRecordCount == 0)
                return false;

            return true;
        }

        public override void OnTrainingDataChanged(string path)
        {
            if (!string.IsNullOrEmpty(path) && File.Exists(path))
            {
                using (FileStream fs = File.OpenRead(path))
                using (var sr = new StreamReader(fs))
                {
                    string line = sr.ReadLine();

                    string[] parts = line.Split(SelectedFieldSeparator.Item2);
                    var fields = new ObservableCollection<FieldDefinitionViewModel>();
                    for (int i = 0; i < parts.Length; i++)
                    {
                        var fld = new FieldDefinitionViewModel
                        {
                            FieldName = FirstLineIsHeader ? parts[i] : string.Format("Column {0}", i),
                            SourceIndex = i,
                            IsEnabled = true,
                            FieldType = FieldTypes.RealValue,
                            FieldWidth = 1,
                            IsLabels = false
                        };

                        fld.FieldChanged += fld_FieldChanged;
                        fields.Add(fld);
                    }

                    int numrows = FirstLineIsHeader ? 0 : 1;
                    while (!sr.EndOfStream)
                    {
                        string ln2 = sr.ReadLine();
                        if (!string.IsNullOrEmpty(ln2))
                            numrows++;
                    }

                    TotalTrainingRecordsAvailableCount = numrows;

                    ObservableCollection<FieldDefinitionViewModel> oldflds = FieldDefinitions;
                    if (oldflds != null)
                    {
                        foreach (FieldDefinitionViewModel fieldDefinitionViewModel in oldflds)
                        {
                            fieldDefinitionViewModel.FieldChanged -= fld_FieldChanged;
                        }
                    }
                    FieldDefinitions = fields;
                    UpdateFieldInfo();
                }
            }
        }

        private void fld_FieldChanged(object sender, EventArgs e)
        {
            UIIsOutOfSync = true;
        }

        private void UpdateFieldInfo()
        {
            UIIsOutOfSync = false;
            var scanners = new IScanner[FieldDefinitions.Count];
            for (int i = 0; i < FieldDefinitions.Count; i++)
            {
                FieldDefinitionViewModel fld = FieldDefinitions[i];
                IScanner scanner = null;
                if (!fld.IsEnabled)
                {
                    scanner = new NullScanner();
                }
                else if (fld.FieldType == FieldTypes.RealValue)
                {
                    scanner = new RealScanner();
                }
                else if (fld.FieldType == FieldTypes.OneOfNOptions)
                {
                    scanner = new OneOfNScanner();
                }
                scanners[i] = scanner;
            }

            foreach (var line in File.ReadAllLines(TrainingDataPath).Skip(FirstLineIsHeader ? 1 : 0))
            {


                if (!string.IsNullOrWhiteSpace(line))
                {
                    string[] parts = line.Split(SelectedFieldSeparator.Item2);
                    Parallel.For(0, scanners.Length, i =>
                        scanners[i].ScanValue(parts[i]));
                }
            }

            for (int i = 0; i < scanners.Length; i++)
            {
                IScanner scanner = scanners[i];
                FieldDefinitionViewModel fld = FieldDefinitions[i];

                if (scanner is NullScanner)
                {
                    fld.MaxRealValue = 0;
                    fld.MinRealValue = 0;
                    fld.OneOfNOptions = null;
                    fld.ParseErrors = 0;
                    fld.FieldWidth = 0;
                }
                else if (scanner is RealScanner)
                {
                    var rs = (RealScanner)scanner;
                    fld.OneOfNOptions = null;
                    fld.MaxRealValue = rs.Min;
                    fld.MaxRealValue = rs.Max;
                    fld.ParseErrors = rs.Errors;
                    fld.FieldWidth = 1;
                }
                else if (scanner is OneOfNScanner)
                {
                    var nScanner = (OneOfNScanner)scanner;
                    fld.MaxRealValue = 0;
                    fld.MinRealValue = 0;
                    fld.OneOfNOptions = nScanner.Options.OrderBy(a => a).ToArray();
                    fld.FieldWidth = fld.UseGrayCodeForOneOfNOptions
                        ? new FieldGrayEncoder<string>(fld.OneOfNOptions).ElementsRequired
                        : nScanner.Options.Count;
                    fld.ParseErrors = nScanner.Errors;
                }

            }
            DataWidth = FieldDefinitions.Where(a => a.IsEnabled && !a.IsLabels).Sum(a => a.FieldWidth);
            LabelWidth = FieldDefinitions.Where(a => a.IsEnabled && a.IsLabels).Sum(a => a.FieldWidth);
            FieldDefinitions = new ObservableCollection<FieldDefinitionViewModel>(FieldDefinitions);//eek
        }

        public override void OnTestDataChanged(string path)
        {
            if (!string.IsNullOrEmpty(path) && File.Exists(path))
            {
                using (FileStream fs = File.OpenRead(path))
                using (var sr = new StreamReader(fs))
                {
                    string line = sr.ReadLine();

                    string[] parts = line.Split(SelectedFieldSeparator.Item2);
                    if (parts.Length == FieldDefinitions.Count - 1)
                        LabelFieldExistsInTestSet = false;

                    else if (FieldDefinitions.Count - parts.Length > 1)
                        throw new Exception();

                    int numrows = FirstLineIsHeader ? 0 : 1;
                    while (!sr.EndOfStream)
                    {
                        string ln2 = sr.ReadLine();
                        if (!string.IsNullOrEmpty(ln2))
                            numrows++;
                    }

                    TotalTestRecordsAvailableCount = numrows;
                }
            }
        }

        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);
            if (e.Property == FirstLineIsHeaderProperty)
            {
                OnTrainingDataChanged(TrainingDataPath);
            }
            //if (e.Property == SelectedLineSeparatorProperty)
            //{
            //    OnTrainingDataChanged(TrainingDataPath);
            //}
            if (e.Property == SelectedFieldSeparatorProperty)
            {
                OnTrainingDataChanged(TrainingDataPath);
                OnTestDataChanged(TestDataPath);
            }
        }

        public override void GetDataReaders(out DataReaderBase<TElement> trainingReader,
            out DataReaderBase<TElement> validationReader, out DataReaderBase<TElement> testReader)
        {
            LineReader<TElement> trainingLabelReader = GetTrainingLabelReader();
            LineReader<TElement> trainingDataReader = GetTrainingDataReader();
            LineReader<TElement> testLabelReader = GetTestLabelReader();
            LineReader<TElement> testDataReader = GetTestDataReader();

            if (testDataReader.DataWidth != trainingDataReader.DataWidth)
            {
                throw new Exception("Mismatch in calculated reader widths");
            }

            if (RandomizeTrainingData)
            {
                trainingReader = new RandomRecordsTextFileReader<TElement>(trainingLabelReader, trainingDataReader,
                    FirstLineIsHeader, TotalTrainingRecordsAvailableCount, TrainingDataPath,
                    SelectedFieldSeparator.Item2);
            }
            else
            {
                trainingReader = new SequentialRecordsTextFileReader<TElement>(trainingLabelReader, trainingDataReader,
                    FirstLineIsHeader, TotalTrainingRecordsAvailableCount, TrainingDataPath,
                    SelectedFieldSeparator.Item2, SkipTrainingRecordCount);
            }

            if (RandomizeValidationData)
            {
                validationReader = new RandomRecordsTextFileReader<TElement>(trainingLabelReader, trainingDataReader,
                    FirstLineIsHeader, TotalTrainingRecordsAvailableCount, TrainingDataPath,
                    SelectedFieldSeparator.Item2);
            }
            else
            {
                validationReader = new SequentialRecordsTextFileReader<TElement>(trainingLabelReader, trainingDataReader,
                    FirstLineIsHeader, TotalTrainingRecordsAvailableCount, TrainingDataPath,
                    SelectedFieldSeparator.Item2, SkipValidationRecordCount);
            }

            if (RandomizeTestData)
            {
                testReader = new RandomRecordsTextFileReader<TElement>(testLabelReader, testDataReader,
                    FirstLineIsHeader, TotalTestRecordsAvailableCount, TestDataPath,
                    SelectedFieldSeparator.Item2);
            }
            else
            {
                testReader = new SequentialRecordsTextFileReader<TElement>(testLabelReader, testDataReader,
                    FirstLineIsHeader, TotalTestRecordsAvailableCount, TestDataPath,
                    SelectedFieldSeparator.Item2, SkipTestRecordCount);
            }
        }

        void GetGlobalDataStats(out TElement minValue, out TElement maxValue)
        {
            TElement min = TElement.MaxValue;
            TElement max = TElement.MinValue;

            foreach (var fieldDefinitionViewModel in FieldDefinitions.Where(a => a.IsEnabled && !a.IsLabels))
            {
                min = Math.Min(fieldDefinitionViewModel.MinRealValue, min);
                max = Math.Max(fieldDefinitionViewModel.MaxRealValue, max);
            }

            minValue = min;
            maxValue = max;
        }

        private LineReader<TElement> GetTrainingLabelReader()
        {
            List<FieldDefinitionViewModel> fields = FieldDefinitions.Where(a => a.IsEnabled && a.IsLabels).ToList();
            var fieldReaders = new List<FieldReaderBase<TElement>>();
            int targetOffset = 0;
            int idx = 0;
            TElement min, max;
            GetGlobalDataStats(out min, out max);
            for (idx = 0; idx < fields.Count; idx++)
            {
                FieldReaderBase<TElement> fr = GetFieldReaderForDefinition(fields[idx], fields[idx].SourceIndex,
                    targetOffset, min, max);
                fieldReaders.Add(fr);
                targetOffset += fr.TargetWidth;
            }

            return new LineReader<TElement>(fieldReaders);
        }

        private LineReader<TElement> GetTestLabelReader()
        {
            if (LabelFieldExistsInTestSet)
            {
                List<FieldDefinitionViewModel> fields = FieldDefinitions.Where(a => a.IsEnabled && a.IsLabels).ToList();
                var fieldReaders = new List<FieldReaderBase<TElement>>();
                int targetOffset = 0;
                int idx = 0;
                TElement min, max;
                GetGlobalDataStats(out min, out max);

                for (idx = 0; idx < fields.Count; idx++)
                {
                    FieldReaderBase<TElement> fr = GetFieldReaderForDefinition(fields[idx], fields[idx].SourceIndex,
                        targetOffset, min, max);
                    fieldReaders.Add(fr);
                    targetOffset += fr.TargetWidth;
                }

                return new LineReader<TElement>(fieldReaders);
            }
            return new LineReader<TElement>(Enumerable.Empty<FieldReaderBase<TElement>>());
        }

        private LineReader<TElement> GetTrainingDataReader()
        {
            List<FieldDefinitionViewModel> fields = FieldDefinitions.Where(a => a.IsEnabled && !a.IsLabels).ToList();
            var fieldReaders = new List<FieldReaderBase<TElement>>();
            int targetOffset = 0;
            int idx = 0;
            TElement min, max;
            GetGlobalDataStats(out min, out max);

            for (idx = 0; idx < fields.Count; idx++)
            {
                FieldReaderBase<TElement> fr = GetFieldReaderForDefinition(fields[idx], fields[idx].SourceIndex,
                    targetOffset, min, max);
                fieldReaders.Add(fr);
                targetOffset += fr.TargetWidth;
            }

            return new LineReader<TElement>(fieldReaders);
        }

        private LineReader<TElement> GetTestDataReader()
        {
            if (LabelFieldExistsInTestSet || FieldDefinitions.Count(a => a.IsEnabled && a.IsLabels) == 0)
            {
                List<FieldDefinitionViewModel> fields = FieldDefinitions.Where(a => a.IsEnabled && !a.IsLabels).ToList();
                var fieldReaders = new List<FieldReaderBase<TElement>>();
                int targetOffset = 0;
                int idx = 0;
                TElement min, max;
                GetGlobalDataStats(out min, out max);

                for (idx = 0; idx < fields.Count; idx++)
                {
                    FieldReaderBase<TElement> fr = GetFieldReaderForDefinition(fields[idx], fields[idx].SourceIndex,
                        targetOffset, min, max);
                    fieldReaders.Add(fr);
                    targetOffset += fr.TargetWidth;
                }

                return new LineReader<TElement>(fieldReaders);
            }
            else
            {
                int sourceIndexModifier = 0;
                int targetOffset = 0;
                var fieldReaders = new List<FieldReaderBase<TElement>>();
                TElement min, max;
                GetGlobalDataStats(out min, out max);

                List<FieldDefinitionViewModel> fields = FieldDefinitions.Where(a => a.IsEnabled).ToList();
                for (int i = 0; i < fields.Count; i++)
                {
                    FieldDefinitionViewModel f = fields[i];
                    if (f.IsEnabled && f.IsLabels)
                    {
                        sourceIndexModifier--;
                        continue;
                    }
                    if (f.IsEnabled)
                    {
                        FieldReaderBase<TElement> fr = GetFieldReaderForDefinition(fields[i],
                            fields[i].SourceIndex + sourceIndexModifier,
                            targetOffset, min, max);
                        fieldReaders.Add(fr);
                        targetOffset += fr.TargetWidth;
                    }
                }
                return new LineReader<TElement>(fieldReaders);
            }
        }

        private FieldReaderBase<TElement> GetFieldReaderForDefinition(FieldDefinitionViewModel fieldDefinitionViewModel,
            int sourceIndex, int targetOffset, TElement globalMinimumValue, TElement globalMaximumValue)
        {
            if (fieldDefinitionViewModel.FieldType == FieldTypes.RealValue)
            {
                //todo:handle any more advance transforms 

                Func<TElement, TElement> convertFromSource = null;
                Func<TElement, TElement> convertToSource = null;

                switch (DataTransformationType)
                {
                    case DataTransformationTypes.NoTransform:
                        {
                            convertFromSource = a => a;
                            convertToSource = a => a;
                            break;
                        }
                    case DataTransformationTypes.DivideBy255:
                        {
                            convertFromSource = a => a / (TElement)255.0;
                            convertToSource = a => a * (TElement)255.0;
                            break;
                        }
                    case DataTransformationTypes.Subtract128Divide127:
                        {
                            convertFromSource = a => (a - (TElement)128.0) / (TElement)127.0;
                            convertToSource = a => (a * (TElement)127.0) + (TElement)128.0;
                            break;
                        }
                    case DataTransformationTypes.DivideByLargestFieldValue:
                    case DataTransformationTypes.DivideByGlobalLargestFieldValue:
                        {
                            var minVal = DataTransformationType.HasFlag(DataTransformationTypes.DivideByGlobalLargestFieldValue)
                                ? globalMinimumValue
                                : fieldDefinitionViewModel.MinRealValue;

                            var maxVal = DataTransformationType.HasFlag(DataTransformationTypes.DivideByGlobalLargestFieldValue)
                                ? globalMaximumValue
                                : fieldDefinitionViewModel.MinRealValue;

                            var divisor = Math.Max(Math.Abs(minVal), Math.Abs(maxVal));

                            if (divisor == 0)
                            {
                                convertFromSource = a => a;
                                convertToSource = a => a;
                            }
                            else
                            {
                                convertFromSource = a => a / divisor;
                                convertToSource = a => a * divisor;
                            }
                            break;
                        }
                }
                return new RealFieldReader<TElement>(sourceIndex, targetOffset, TElement.Parse, convertFromSource, convertToSource);
            }
            if (fieldDefinitionViewModel.UseGrayCodeForOneOfNOptions)
            {
                return new OneOfNOptionsGrayCodedFieldReader<TElement>(sourceIndex, targetOffset,
                    fieldDefinitionViewModel.OneOfNOptions, (TElement)1.0, (TElement)0.0, a => a, a => a);
            }
            return new OneOfNOptionsFieldReader<TElement>(sourceIndex, targetOffset,
                fieldDefinitionViewModel.OneOfNOptions, (TElement)1.0, (TElement)0.0, a => a, a => a);
        }


        private interface IScanner
        {
            int Errors { get; }
            void ScanValue(string s);
        }

        private class NullScanner : IScanner
        {
            public int Errors
            {
                get { return 0; }
            }

            public void ScanValue(string s)
            {
                //do nothing
            }
        }

        private class OneOfNScanner : IScanner
        {
            private HashSet<string> _options = new HashSet<string>();


            public HashSet<string> Options
            {
                get { return _options; }
                private set { _options = value; }
            }

            public int Errors { get; set; }

            public void ScanValue(string s)
            {
                try
                {
                    _options.Add(s.Trim());
                }
                catch (Exception ex)
                {
                    Errors++;
                }
            }
        }

        private class RealScanner : IScanner
        {
            private TElement _max = TElement.MinValue;
            private TElement _min = TElement.MaxValue;

            public TElement Min
            {
                get { return _min; }
                private set { _min = value; }
            }

            public TElement Max
            {
                get { return _max; }
                private set { _max = value; }
            }

            public int Errors { get; private set; }

            public void ScanValue(string value)
            {
                if (string.IsNullOrWhiteSpace(value))
                    return;
                TElement d;
                if (TElement.TryParse(value.Trim(), out d))
                {
                    Min = Math.Min(Min, d);
                    Max = Math.Max(Max, d);
                }
                else
                {
                    Errors++;
                }
            }
        }
    }
}