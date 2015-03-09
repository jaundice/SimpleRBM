using System;
using System.IO;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Input;
using CudaNN.DeepBelief.DataIO;
using OpenFileDialog = Microsoft.Win32.OpenFileDialog;

namespace CudaNN.DeepBelief.ViewModels
{
    public abstract class DataConfigViewModelBase : DependencyObject
    {
        public enum DataContainerType
        {
            Directory,
            File
        }

        public enum DataTransformationTypes
        {
            NoTransform,
            DivideBy255,
            Subtract128Divide127,
            DivideByGlobalLargestFieldValue,
            DivideByLargestFieldValue
        }

        public enum NetworkUsageTypes
        {
            SupervisedLabellingNetwork,
            UnsupervisedCodingNetwork,
        }


        public static readonly DependencyProperty NetworkUsageTypeProperty =
          DependencyProperty.Register("NetworkUsageType", typeof(NetworkUsageTypes), typeof(DataConfigViewModelBase),
              new PropertyMetadata(NetworkUsageTypes.UnsupervisedCodingNetwork));

        public static readonly DependencyProperty DataTransformationTypeProperty =
            DependencyProperty.Register("DataTransformationType", typeof(DataTransformationTypes), typeof(DataConfigViewModelBase),
                new PropertyMetadata(DataTransformationTypes.NoTransform));

        public static readonly DependencyProperty BrowseTrainingDataCommandProperty =
         DependencyProperty.Register("BrowseTrainingDataCommand", typeof(ICommand), typeof(DataConfigViewModelBase),
             new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty BrowseTestDataCommandProperty =
            DependencyProperty.Register("BrowseTestDataCommand", typeof(ICommand), typeof(DataConfigViewModelBase),
                new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty TrainingDataPathProperty =
            DependencyProperty.Register("TrainingDataPath", typeof(string),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(string)));

        public static readonly DependencyProperty TestDataPathProperty = DependencyProperty.Register("TestDataPath",
            typeof(string),
            typeof(DataConfigViewModelBase), new PropertyMetadata(default(string)));


        public static readonly DependencyProperty RandomizeTrainingDataProperty =
            DependencyProperty.Register("RandomizeTrainingData",
                typeof(bool),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(bool)));

        public static readonly DependencyProperty RandomizeValidationDataProperty =
            DependencyProperty.Register("RandomizeValidationData",
                typeof(bool),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(bool)));

        public static readonly DependencyProperty RandomizeTestDataProperty =
            DependencyProperty.Register("RandomizeTestData",
                typeof(bool),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(bool)));

        public static readonly DependencyProperty TrainingRecordCountProperty =
            DependencyProperty.Register("TrainingRecordCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));


        public static readonly DependencyProperty SkipTrainingRecordCountProperty =
            DependencyProperty.Register("SkipTrainingRecordCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));


        public static readonly DependencyProperty ValidationRecordCountProperty =
            DependencyProperty.Register("ValidationRecordCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));


        public static readonly DependencyProperty SkipValidationRecordCountProperty =
            DependencyProperty.Register("SkipValidationRecordCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));


        public static readonly DependencyProperty TotalTrainingRecordsAvailableCountProperty =
            DependencyProperty.Register("TotalTrainingRecordsAvailableCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));


        public static readonly DependencyProperty TestRecordCountProperty =
            DependencyProperty.Register("TestRecordCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty SkipTestRecordCountProperty =
            DependencyProperty.Register("SkipTestRecordCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));


        public static readonly DependencyProperty TotalTestRecordsAvailableCountProperty =
            DependencyProperty.Register("TotalTestRecordsAvailableCount",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty DataWidthProperty =
            DependencyProperty.Register("DataWidth",
                typeof(int),
                typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));
        public static readonly DependencyProperty LabelWidthProperty =
            DependencyProperty.Register("LabelWidth",
            typeof(int),
            typeof(DataConfigViewModelBase), new PropertyMetadata(default(int)));

        public abstract void GetDataReaders(out DataReaderBase<double> trainingReader,
            out DataReaderBase<double> validationReader, out DataReaderBase<double> testReader);


        public NetworkUsageTypes NetworkUsageType
        {
            get { return (NetworkUsageTypes)GetValue(NetworkUsageTypeProperty); }
            set { SetValue(NetworkUsageTypeProperty, value); }
        }



        public DataTransformationTypes DataTransformationType
        {
            get { return (DataTransformationTypes)GetValue(DataTransformationTypeProperty); }
            set { SetValue(DataTransformationTypeProperty, value); }
        }

        public NetworkUsageTypes[] AllNetworkUsageTypes
        {
            get
            {
                return new[]
                {
                    NetworkUsageTypes.UnsupervisedCodingNetwork,
                    NetworkUsageTypes.SupervisedLabellingNetwork
                };
            }
        }

        public bool RandomizeTrainingData
        {
            get { return (bool)GetValue(RandomizeTrainingDataProperty); }
            set { SetValue(RandomizeTrainingDataProperty, value); }
        }

        public bool RandomizeValidationData
        {
            get { return (bool)GetValue(RandomizeValidationDataProperty); }
            set { SetValue(RandomizeValidationDataProperty, value); }
        }

        public bool RandomizeTestData
        {
            get { return (bool)GetValue(RandomizeTestDataProperty); }
            set { SetValue(RandomizeTestDataProperty, value); }
        }

        public string TrainingDataPath
        {
            get { return (string)GetValue(TrainingDataPathProperty); }
            set { SetValue(TrainingDataPathProperty, value); }
        }

        public int TrainingRecordCount
        {
            get { return (int)GetValue(TrainingRecordCountProperty); }
            set { SetValue(TrainingRecordCountProperty, value); }
        }

        public int SkipTrainingRecordCount
        {
            get { return (int)GetValue(SkipTrainingRecordCountProperty); }
            set { SetValue(SkipTrainingRecordCountProperty, value); }
        }


        public int ValidationRecordCount
        {
            get { return (int)GetValue(ValidationRecordCountProperty); }
            set { SetValue(ValidationRecordCountProperty, value); }
        }

        public int SkipValidationRecordCount
        {
            get { return (int)GetValue(SkipValidationRecordCountProperty); }
            set { SetValue(SkipValidationRecordCountProperty, value); }
        }

        public int TotalTrainingRecordsAvailableCount
        {
            get { return (int)GetValue(TotalTrainingRecordsAvailableCountProperty); }
            set { SetValue(TotalTrainingRecordsAvailableCountProperty, value); }
        }

        public string TestDataPath
        {
            get { return (string)GetValue(TestDataPathProperty); }
            set { SetValue(TestDataPathProperty, value); }
        }

        public int TestRecordCount
        {
            get { return (int)GetValue(TestRecordCountProperty); }
            set { SetValue(TestRecordCountProperty, value); }
        }

        public int SkipTestRecordCount
        {
            get { return (int)GetValue(SkipTestRecordCountProperty); }
            set { SetValue(SkipTestRecordCountProperty, value); }
        }

        public int TotalTestRecordsAvailableCount
        {
            get { return (int)GetValue(TotalTestRecordsAvailableCountProperty); }
            set { SetValue(TotalTestRecordsAvailableCountProperty, value); }
        }

        public int DataWidth
        {
            get { return (int)GetValue(DataWidthProperty); }
            set { SetValue(DataWidthProperty, value); }
        }
        public int LabelWidth
        {
            get { return (int)GetValue(LabelWidthProperty); }
            set { SetValue(LabelWidthProperty, value); }
        }
        public ICommand BrowseTrainingDataCommand
        {
            get { return (ICommand)GetValue(BrowseTrainingDataCommandProperty); }
            set { SetValue(BrowseTrainingDataCommandProperty, value); }
        }

        public ICommand BrowseTestDataCommand
        {
            get { return (ICommand)GetValue(BrowseTestDataCommandProperty); }
            set { SetValue(BrowseTestDataCommandProperty, value); }
        }

        public abstract DataContainerType ContainerType { get; }

        public virtual bool Validate()
        {
            Func<string, bool> pathValidator = this.ContainerType == DataContainerType.Directory
                ? (Func<string, bool>)(Directory.Exists)
                : (Func<string, bool>)(File.Exists);

            if (string.IsNullOrWhiteSpace(TrainingDataPath) || string.IsNullOrWhiteSpace(TestDataPath))
                return false;

            if (!pathValidator(TrainingDataPath) || !pathValidator(TestDataPath))
                return false;

            if (TrainingRecordCount == 0 || TestRecordCount == 0)
                return false;

            if (SkipTrainingRecordCount + TrainingRecordCount > TotalTrainingRecordsAvailableCount)
                return false;

            if (SkipTestRecordCount + TestRecordCount > TotalTestRecordsAvailableCount)
                return false;

            if (SkipValidationRecordCount + ValidationRecordCount > TotalTrainingRecordsAvailableCount)
                return false;

            if (NetworkUsageType == NetworkUsageTypes.SupervisedLabellingNetwork && LabelWidth == 0)
                return false;

            return true;
        }

        public abstract string FileExtensionFilter { get; }

        public string BrowseForData()
        {
            switch (ContainerType)
            {
                case DataContainerType.Directory:
                    {
                        using (var browser = new FolderBrowserDialog())
                        {
                            browser.Description = "Browse for directory containing the data";
                            browser.ShowNewFolderButton = false;
                            return browser.ShowDialog() == DialogResult.OK ? browser.SelectedPath : null;
                        }
                    }
                case DataContainerType.File:
                    {
                        var ofd = new OpenFileDialog()
                        {
                            CheckPathExists = true,
                            Multiselect = false,
                            Title = "Browse for data file",
                            Filter = FileExtensionFilter,
                        };

                        var ret = ofd.ShowDialog(Window.GetWindow(this));
                        return ret.Value ? ofd.FileName : null;
                    }
                default:
                    throw new NotImplementedException();
            }
        }

        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);
            if (e.Property == TrainingDataPathProperty)
            {
                OnTrainingDataChanged((string)e.NewValue);
            }
            if (e.Property == TestDataPathProperty)
            {
                OnTestDataChanged((string)e.NewValue);
            }
            if (e.Property == RandomizeTrainingDataProperty)
            {
                if ((bool)e.NewValue)
                    SkipTrainingRecordCount = 0;
            }
            if (e.Property == RandomizeValidationDataProperty)
            {
                if ((bool)e.NewValue)
                    SkipValidationRecordCount = 0;
            }
            if (e.Property == RandomizeTestDataProperty)
            {
                if ((bool)e.NewValue)
                    SkipTestRecordCount = 0;
            }
        }

        public abstract void OnTrainingDataChanged(string path);

        public abstract void OnTestDataChanged(string path);

        protected DataConfigViewModelBase()
        {
            BrowseTrainingDataCommand = new CommandHandler(a => { TrainingDataPath = BrowseForData(); }, a => true);

            BrowseTestDataCommand = new CommandHandler(a => { TestDataPath = BrowseForData(); }, a => true);
        }
    }
}