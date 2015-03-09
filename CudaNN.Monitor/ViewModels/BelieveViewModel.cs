using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using CudaNN.DeepBelief.DataIO;
using CudaNN.DeepBelief.LayerBuilders;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Cuda;
using SimpleRBM.Demo;
using Brush = System.Windows.Media.Brush;
using Image = System.Windows.Controls.Image;
using Point = System.Windows.Point;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;

#endif

namespace CudaNN.DeepBelief.ViewModels
{
    public class BelieveViewModel : DependencyObject, IDisposable
    {
        public static readonly DependencyProperty DemoModeProperty = DependencyProperty.Register("DemoMode",
            typeof(string), typeof(BelieveViewModel), new PropertyMetadata(default(string)));

        public static readonly DependencyProperty LayerProperty = DependencyProperty.Register("Layer", typeof(int),
            typeof(BelieveViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty SelectedFeatureIndexProperty =
            DependencyProperty.Register("SelectedFeatureIndex", typeof(int),
                typeof(BelieveViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty BatchSizeProperty =
            DependencyProperty.Register("BatchSize", typeof(int),
                typeof(BelieveViewModel), new PropertyMetadata(100));

        public static readonly DependencyProperty DefaultSuspendStateProperty =
            DependencyProperty.Register("DefaultSuspendState", typeof(SuspendState),
                typeof(BelieveViewModel), new PropertyMetadata(SuspendState.Active));


        public static readonly DependencyProperty KeepDataInSystemMemoryProperty =
            DependencyProperty.Register("KeepDataInSystemMemory", typeof(bool),
                typeof(BelieveViewModel), new PropertyMetadata(default(bool)));


        public static readonly DependencyProperty UpdateActivationsEveryEpochProperty =
            DependencyProperty.Register("UpdateActivationsEveryEpoch", typeof(bool),
                typeof(BelieveViewModel), new PropertyMetadata(true));


        public static readonly DependencyProperty EpochProperty = DependencyProperty.Register("Epoch", typeof(int),
            typeof(BelieveViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty LearningRateProperty = DependencyProperty.Register("LearningRate",
            typeof(Double), typeof(BelieveViewModel), new PropertyMetadata(default(Double)));

        public static readonly DependencyProperty UpdateFrequencyProperty =
            DependencyProperty.Register("UpdateFrequency", typeof(int), typeof(BelieveViewModel),
                new PropertyMetadata(200));

        public static readonly DependencyProperty ActivationFrequencyProperty =
            DependencyProperty.Register("ActivationFrequency", typeof(BitmapSource), typeof(BelieveViewModel),
                new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty ReconstructionsProperty =
            DependencyProperty.Register("Reconstructions", typeof(ObservableCollection<ValidationSet>),
                typeof(BelieveViewModel), new PropertyMetadata(default(ObservableCollection<ValidationSet>)));

        public static readonly DependencyProperty FeaturesProperty = DependencyProperty.Register("Features",
            typeof(ObservableCollection<BitmapSource>), typeof(BelieveViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty RunAppMethodProperty = DependencyProperty.Register("RunAppMethod",
            typeof(ICommand), typeof(BelieveViewModel), new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty RunBindingProperty = DependencyProperty.Register("RunBinding",
            typeof(CommandBinding), typeof(BelieveViewModel), new PropertyMetadata(default(CommandBinding)));

        public static readonly DependencyProperty RunCommandProperty = DependencyProperty.Register("RunCommand",
            typeof(ICommand), typeof(BelieveViewModel), new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty ScaleLearningRatesCommandProperty =
            DependencyProperty.Register("ScaleLearningRatesCommandCommand",
                typeof(ICommand), typeof(BelieveViewModel), new PropertyMetadata(default(ICommand)));


        public static readonly DependencyProperty BackupFrequencyProperty =
            DependencyProperty.Register("BackupFrequency", typeof(int), typeof(BelieveViewModel),
                new PropertyMetadata(1000));

        public static readonly DependencyProperty ErrorProperty = DependencyProperty.Register("Error", typeof(Double),
            typeof(BelieveViewModel), new PropertyMetadata(default(Double)));

        public static readonly DependencyProperty DeltaProperty = DependencyProperty.Register("Delta", typeof(Double),
            typeof(BelieveViewModel), new PropertyMetadata(default(Double)));

        public static readonly DependencyProperty NumTrainingExamplesProperty =
            DependencyProperty.Register("NumTrainingExamples", typeof(int), typeof(BelieveViewModel),
                new PropertyMetadata(1000));

        public static readonly DependencyProperty DayDreamsProperty = DependencyProperty.Register("DayDreams",
            typeof(ObservableCollection<BitmapSource>), typeof(BelieveViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty TrainingSetProperty = DependencyProperty.Register("TrainingSet",
            typeof(ObservableCollection<BitmapSource>), typeof(BelieveViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty ElapsedProperty = DependencyProperty.Register("Elapsed",
            typeof(TimeSpan), typeof(BelieveViewModel), new PropertyMetadata(default(TimeSpan)));

        public static readonly DependencyProperty ExitEvaluatorFactoryProperty =
            DependencyProperty.Register("ExitEvaluatorFactory", typeof(InteractiveExitEvaluatorFactory<double>),
                typeof(BelieveViewModel), new PropertyMetadata(default(InteractiveExitEvaluatorFactory<double>)));

        public static readonly DependencyProperty DisplayedEpochProperty = DependencyProperty.Register(
            "DisplayedEpoch", typeof(int), typeof(BelieveViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty WeightLearningRateFactoryProperty =
            DependencyProperty.Register("WeightLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<double>), typeof(BelieveViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<double>)));

        public static readonly DependencyProperty DisplayFeatureCommandProperty =
            DependencyProperty.Register("DisplayFeatureCommand", typeof(ICommand), typeof(BelieveViewModel),
                new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty SelectedFeatureProperty =
            DependencyProperty.Register("SelectedFeature", typeof(BitmapSource), typeof(BelieveViewModel),
                new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty VisBiasLearningRateFactoryProperty =
            DependencyProperty.Register("VisBiasLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<double>), typeof(BelieveViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<double>)));

        public static readonly DependencyProperty HidBiasLearningRateFactoryProperty =
            DependencyProperty.Register("HidBiasLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<double>), typeof(BelieveViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<double>)));

        public static readonly DependencyProperty ErrorLabelBrushProperty =
            DependencyProperty.Register("ErrorLabelBrush", typeof(Brush), typeof(BelieveViewModel),
                new PropertyMetadata(default(Brush)));

        public static readonly DependencyProperty DeltaLabelBrushProperty =
            DependencyProperty.Register("DeltaLabelBrush", typeof(Brush), typeof(BelieveViewModel),
                new PropertyMetadata(default(Brush)));

        public static readonly DependencyProperty LayerConfigsProperty = DependencyProperty.Register("LayerConfigs",
            typeof(ObservableCollection<ConstructLayerBase>), typeof(BelieveViewModel),
            new PropertyMetadata(default(ObservableCollection<ConstructLayerBase>)));


        private CancellationTokenSource _cancelSource;
        private Task _runTask;


        public BelieveViewModel()
        {
            RunCommand = new CommandHandler(a => Run(), a => true);
            ScaleLearningRatesCommand = new CommandHandler(a =>
            {
                double rate = Convert.ToDouble(a);
                WeightLearningRateFactory.InnerCalculators[Layer].LearningRate *= rate;
                HidBiasLearningRateFactory.InnerCalculators[Layer].LearningRate *= rate;
                VisBiasLearningRateFactory.InnerCalculators[Layer].LearningRate *= rate;
            }, a => true);
        }

        public SuspendState[] AllSuspendStates
        {
            get { return new[] { SuspendState.Active, SuspendState.Suspended }; }
        }

        public string DemoMode
        {
            get { return Dispatcher.InvokeIfRequired(() => (string)GetValue(DemoModeProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DemoModeProperty, value)).Wait(); }
        }

        public int BatchSize
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(BatchSizeProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(BatchSizeProperty, value)).Wait(); }
        }

        public bool KeepDataInSystemMemory
        {
            get { return Dispatcher.InvokeIfRequired(() => (bool)GetValue(KeepDataInSystemMemoryProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(KeepDataInSystemMemoryProperty, value)).Wait(); }
        }


        public SuspendState DefaultSuspendState
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (SuspendState)GetValue(DefaultSuspendStateProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DefaultSuspendStateProperty, value)).Wait(); }
        }

        public int Layer
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(LayerProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LayerProperty, value)).Wait(); }
        }

        public int SelectedFeatureIndex
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(SelectedFeatureIndexProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(SelectedFeatureIndexProperty, value)).Wait(); }
        }

        public int Epoch
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(EpochProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(EpochProperty, value)).Wait(); }
        }

        public Double LearningRate
        {
            get { return Dispatcher.InvokeIfRequired(() => (Double)GetValue(LearningRateProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LearningRateProperty, value)).Wait(); }
        }

        public bool UpdateActivationsEveryEpoch
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (bool)GetValue(UpdateActivationsEveryEpochProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(UpdateActivationsEveryEpochProperty, value)).Wait(); }
        }

        public int UpdateFrequency
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(UpdateFrequencyProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(UpdateFrequencyProperty, value)).Wait(); }
        }

        public BitmapSource ActivationFrequency
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (BitmapSource)GetValue(ActivationFrequencyProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ActivationFrequencyProperty, value)).Wait(); }
        }

        public ObservableCollection<ValidationSet> Reconstructions
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () => (ObservableCollection<ValidationSet>)GetValue(ReconstructionsProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ReconstructionsProperty, value)).Wait(); }
        }

        public ObservableCollection<BitmapSource> Features
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (ObservableCollection<BitmapSource>)GetValue(FeaturesProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(FeaturesProperty, value)).Wait(); }
        }

        public ICommand RunCommand
        {
            get { return Dispatcher.InvokeIfRequired(() => (ICommand)GetValue(RunCommandProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(RunCommandProperty, value)).Wait(); }
        }

        public ICommand ScaleLearningRatesCommand
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (ICommand)GetValue(ScaleLearningRatesCommandProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ScaleLearningRatesCommandProperty, value)).Wait(); }
        }

        public int BackupFrequency
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(BackupFrequencyProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(BackupFrequencyProperty, value)); }
        }

        public Double Error
        {
            get { return Dispatcher.InvokeIfRequired(() => (Double)GetValue(ErrorProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ErrorProperty, value)).Wait(); }
        }

        public Double Delta
        {
            get { return Dispatcher.InvokeIfRequired(() => (Double)GetValue(DeltaProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DeltaProperty, value)).Wait(); }
        }

        public int NumTrainingExamples
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(NumTrainingExamplesProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(NumTrainingExamplesProperty, value)).Wait(); }
        }

        public ObservableCollection<BitmapSource> DayDreams
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (ObservableCollection<BitmapSource>)GetValue(DayDreamsProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DayDreamsProperty, value)).Wait(); }
        }

        public ObservableCollection<BitmapSource> TrainingSet
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (ObservableCollection<BitmapSource>)GetValue(TrainingSetProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(TrainingSetProperty, value)).Wait(); }
        }

        public TimeSpan Elapsed
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (TimeSpan)GetValue(ElapsedProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ElapsedProperty, value)).Wait(); }
        }

        public InteractiveExitEvaluatorFactory<double> ExitEvaluatorFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () => (InteractiveExitEvaluatorFactory<double>)GetValue(ExitEvaluatorFactoryProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ExitEvaluatorFactoryProperty, value)).Wait(); }
        }

        public int DisplayedEpoch
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(DisplayedEpochProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DisplayedEpochProperty, value)).Wait(); }
        }

        public LayerSpecificLearningRateCalculatorFactory<double> WeightLearningRateFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () =>
                            (LayerSpecificLearningRateCalculatorFactory<double>)
                                GetValue(WeightLearningRateFactoryProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(WeightLearningRateFactoryProperty, value)).Wait(); }
        }

        public ICommand DisplayFeatureCommand
        {
            get { return Dispatcher.InvokeIfRequired(() => (ICommand)GetValue(DisplayFeatureCommandProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DisplayFeatureCommandProperty, value)).Wait(); }
        }

        public BitmapSource SelectedFeature
        {
            get { return Dispatcher.InvokeIfRequired(() => (BitmapSource)GetValue(SelectedFeatureProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(SelectedFeatureProperty, value)).Wait(); }
        }

        public LayerSpecificLearningRateCalculatorFactory<double> VisBiasLearningRateFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () =>
                            (LayerSpecificLearningRateCalculatorFactory<double>)
                                GetValue(VisBiasLearningRateFactoryProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(VisBiasLearningRateFactoryProperty, value)).Wait(); }
        }

        public LayerSpecificLearningRateCalculatorFactory<double> HidBiasLearningRateFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () =>
                            (LayerSpecificLearningRateCalculatorFactory<double>)
                                GetValue(HidBiasLearningRateFactoryProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(HidBiasLearningRateFactoryProperty, value)).Wait(); }
        }

        public Brush ErrorLabelBrush
        {
            get { return Dispatcher.InvokeIfRequired(() => (Brush)GetValue(ErrorLabelBrushProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ErrorLabelBrushProperty, value)).Wait(); }
        }

        public Brush DeltaLabelBrush
        {
            get { return Dispatcher.InvokeIfRequired(() => (Brush)GetValue(DeltaLabelBrushProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DeltaLabelBrushProperty, value)).Wait(); }
        }

        public bool IsDisposed { get; protected set; }

        public ObservableCollection<ConstructLayerBase> LayerConfigs
        {
            get { return (ObservableCollection<ConstructLayerBase>)GetValue(LayerConfigsProperty); }
            set { SetValue(LayerConfigsProperty, value); }
        }

        public void Dispose()
        {
            if (IsDisposed)
                return;
            IsDisposed = true;
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void DisplayFeature(object sender, MouseEventArgs mouseArgs)
        {
            if (mouseArgs != null)
            {
                var image = mouseArgs.Device.Target as Image;
                Point pos = mouseArgs.GetPosition(image);

                var imgSrc = (BitmapSource)image.Source;
                double pixX = pos.X * imgSrc.PixelWidth / image.ActualWidth;
                double pixY = pos.Y * imgSrc.PixelHeight / image.ActualHeight;

                int idx = (int)(pixY) * imgSrc.PixelWidth + (int)pixX;
                SelectedFeatureIndex = idx;
                SelectFeature(idx);
            }
        }

        private void SelectFeature(int idx)
        {
            if (Features != null)
            {
                SelectedFeature = idx < Features.Count ? Features[idx] : null;
            }
        }


        private async void Run()
        {
            string pathBase = Path.Combine(Environment.CurrentDirectory,
                string.Format("{0}_{1}", DateTime.Now.ToString("u").Replace(':', '-'), DemoMode));

            Directory.CreateDirectory(pathBase);
            UpdateFrequency = UpdateFrequency > 0 ? UpdateFrequency : (UpdateFrequency = 20);
            BackupFrequency = BackupFrequency > 0 ? BackupFrequency : (BackupFrequency = 500);

            Task t = null;
            try
            {
                if (_cancelSource != null)
                    _cancelSource.Cancel();
                if (_runTask != null)
                    await _runTask;
            }
            catch (TaskCanceledException)
            {
            }
            catch (OperationCanceledException)
            {
            }

            _cancelSource = new CancellationTokenSource();

            try
            {
                ConfigureData data;
                if (!TryConfigureData(out data)) return;

                LayerBuilderViewModel builder;
                if (
                    !TryConfigureNetwork(out builder, ((DataConfigViewModelBase)data.DataContext).DataWidth,
                        ((DataConfigViewModelBase)data.DataContext).LabelWidth)) return;

                ConfigureLearningRates(builder);

                var configRun = new ConfigureRun
                {
                    Owner = Window.GetWindow(this),
                    DataContext = this,
                    WindowStyle = WindowStyle.None,
                    WindowStartupLocation = WindowStartupLocation.CenterOwner
                };
                configRun.ShowDialog();

                var dataDc = (DataConfigViewModelBase)data.DataContext;
                int batchSize = BatchSize;
                bool useSysMemory = KeepDataInSystemMemory;
                SuspendState susState = DefaultSuspendState;
                if (!useSysMemory)
                {
                    _runTask = Task.Run(() => ExecuteWithGPUMemory(builder, dataDc, batchSize, susState, pathBase),
                        _cancelSource.Token);
                }
                else
                {
                    _runTask = Task.Run(() => ExecuteWithSystemMemory(builder, dataDc, batchSize, susState, pathBase));
                }
            }
            catch (TaskCanceledException)
            {
            }
            catch (OperationCanceledException)
            {
            }
        }

        private void ConfigureLearningRates(LayerBuilderViewModel builder)
        {
            WeightLearningRateFactory =
                new LayerSpecificLearningRateCalculatorFactory<double>(
                    builder.LayerConstructionInfo.Select(
                        a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
            HidBiasLearningRateFactory =
                new LayerSpecificLearningRateCalculatorFactory<double>(
                    builder.LayerConstructionInfo.Select(
                        a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
            VisBiasLearningRateFactory =
                new LayerSpecificLearningRateCalculatorFactory<double>(
                    builder.LayerConstructionInfo.Select(
                        a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
            var lrEditor = new ConfigureLearningRates
            {
                Owner = Window.GetWindow(this),
                WindowStyle = WindowStyle.None,
                WindowStartupLocation = WindowStartupLocation.CenterOwner,
                DataContext = this
            };
            lrEditor.ShowDialog();
        }

        private bool TryConfigureNetwork(out LayerBuilderViewModel builder, int dataWidth, int labelWidth)
        {
            var defineDlg = new DefineNetwork
            {
                Owner = Window.GetWindow(this),
                WindowStartupLocation = WindowStartupLocation.CenterOwner,
                WindowStyle = WindowStyle.None
            };
            builder = defineDlg.DataContext as LayerBuilderViewModel;
            LayerConfigs = builder.LayerConstructionInfo;


            switch (DemoMode)
            {
                case "Faces":
                    {
                        ConfigureDefaultFacesLayers(builder, dataWidth, labelWidth);
                        break;
                    }
                case "Data":
                    {
                        ConfigureDefaultDataLayers(builder, dataWidth, labelWidth);
                        break;
                    }
                case "Kaggle":
                    {
                        ConfigureDefaultKaggleLayers(builder, dataWidth, labelWidth);
                        break;
                    }
            }

            bool? rr = defineDlg.ShowDialog();
            if (!rr.HasValue || !rr.Value)
                return false;
            return true;
        }

        private bool TryConfigureData(out ConfigureData data)
        {
            data = new ConfigureData
            {
                Owner = Window.GetWindow(this),
                WindowStartupLocation = WindowStartupLocation.CenterOwner,
                WindowStyle = WindowStyle.None
            };
            bool? res = data.ShowDialog();
            if (!res.HasValue || !res.Value)
                return false;
            return true;
        }

        private void ConfigureDefaultKaggleLayers(LayerBuilderViewModel builderViewModel, int dataWidth, int labelWidth)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = dataWidth,
                NumHiddenNeurons = 500,
                ConvertActivationsToStates = false,
                WeightInitializationStDev = 0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500,
                NumHiddenNeurons = 500,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = 0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500 + labelWidth,
                NumHiddenNeurons = 2000,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = 0.01
            });
        }

        private void ConfigureDefaultDataLayers(LayerBuilderViewModel builderViewModel, int dataWidth, int labelWidth)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = dataWidth,
                NumHiddenNeurons = 500,
                ConvertActivationsToStates = false,
                WeightInitializationStDev = 0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500,
                NumHiddenNeurons = 500,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = 0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500 + labelWidth,
                NumHiddenNeurons = 50,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = 0.01
            });
        }

        private void ConfigureDefaultFacesLayers(LayerBuilderViewModel builderViewModel, int dataWidth, int labelWidth)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructLinearHiddenLayer
            {
                NumVisibleNeurons = dataWidth,
                NumHiddenNeurons = 2000,
                WeightInitializationStDev = 0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructLinearHiddenLayer
            {
                NumVisibleNeurons = 2000,
                NumHiddenNeurons = 4000,
                WeightInitializationStDev = 0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructLinearHiddenLayer
            {
                NumVisibleNeurons = 4000 + labelWidth,
                NumHiddenNeurons = 4000,
                WeightInitializationStDev = 0.01
            });
        }


        private async void ExecuteWithGPUMemory(LayerBuilderViewModel layerBuilderViewModel,
            DataConfigViewModelBase dataConfigViewModel, int batchSize, SuspendState defaultSuspendState,
            string pathBase)
        {
            using (
                var greedyTracker =
                    new EpochErrorFileTracker<double>(Path.Combine(pathBase, "GreedyTrainError.log")))
            {
                int validationRecords = 0, trainingRecords = 0, testRecords = 0;
                var usageType = DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork;
                var dtType = DataConfigViewModelBase.DataTransformationTypes.NoTransform;
                DataReaderBase<double> testReader = null;
                DataReaderBase<double> trainingReader = null;
                DataReaderBase<double> validationReader = null;
                await Dispatcher.InvokeIfRequired(() =>
                {
                    dtType = dataConfigViewModel.DataTransformationType;
                    usageType = dataConfigViewModel.NetworkUsageType;
                    validationRecords = dataConfigViewModel.ValidationRecordCount;
                    trainingRecords = dataConfigViewModel.TrainingRecordCount;
                    testRecords = dataConfigViewModel.TestRecordCount;
                    dataConfigViewModel.GetDataReaders(out trainingReader, out validationReader, out testReader);
                });

                Func<double[,], Task<IList<BitmapSource>>> imageFactory = dtType ==
                                                                          DataConfigViewModelBase
                                                                              .DataTransformationTypes
                                                                              .Subtract128Divide127
                    ? (Func<double[,], Task<IList<BitmapSource>>>)(dd => GenerateImageSourcesPosNeg(dd))
                    : dd => GenerateImageSources(dd);


                string[] validationLabels;
                Double[,] validationLabelsCoded;
                double[,] validationData = validationReader.ReadWithLabels(validationRecords, out validationLabelsCoded,
                    out validationLabels);
                Task<IList<BitmapSource>> validationImages = imageFactory(validationData);


                string[] trainingLabels;
                Double[,] traingLabelsCoded;
                double[,] trainingData = trainingReader.ReadWithLabels(trainingRecords, out traingLabelsCoded,
                    out trainingLabels);

                Dispatcher.InvokeIfRequired(
                    async () =>
                        TrainingSet =
                            new ObservableCollection<BitmapSource>(await imageFactory(trainingData)));

                if (usageType == DataConfigViewModelBase.NetworkUsageTypes.SupervisedLabellingNetwork)
                {
                    Task<IList<BitmapSource>> validationCodes = GenerateImageSources(validationLabelsCoded);
                    Dispatcher.InvokeIfRequired(
                        async () =>
                            Reconstructions =
                                new ObservableCollection<ValidationSet>((await validationImages).Zip(
                                    await validationCodes,
                                    (source, bitmapSource) =>
                                        new ValidationSet
                                        {
                                            OriginalImageSet = new ImageSet
                                            {
                                                DataImage = source,
                                                CodeImage = bitmapSource
                                            },
                                            ReconstructedImageSet = new ImageSet()
                                        })
                                    .Zip(validationLabels,
                                        (set, s) =>
                                        {
                                            set.OriginalImageSet.Label = s;
                                            return set;
                                        })));
                }
                else
                {
                    Dispatcher.InvokeIfRequired(
                        async () =>
                            Reconstructions =
                                new ObservableCollection<ValidationSet>(
                                    (await validationImages).Select(a => new ValidationSet
                                    {
                                        OriginalImageSet = new ImageSet
                                        {
                                            DataImage = a
                                        },
                                        ReconstructedImageSet = new ImageSet()
                                    })));
                }


                await Dispatcher.InvokeIfRequired(() =>
                {
                    ExitEvaluatorFactory = new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000);
                    NumTrainingExamples = trainingData.GetLength(0);
                });

                GPGPU dev;
                GPGPURAND rand;
                InitCuda(out dev, out rand);
                dev.SetCurrentContext();
                using (var net = new CudaAdvancedNetwork(layerBuilderViewModel.CreateLayers(dev, rand)))
                {
                    List<double[,]> identityMatrices = IdentityMatrices(dev, net);
                    net.SetDefaultMachineState(defaultSuspendState);
                    dev.SetCurrentContext();

                    net.EpochComplete += usageType ==
                                         DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork
                        ? NetEpochUnsupervisedCompleteEventHandler(pathBase, dev, validationData, identityMatrices,
                            imageFactory)
                        : NetEpochSupervisedCompleteEventHandler(pathBase, validationData, identityMatrices, dev,
                            imageFactory, validationReader);

                    net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

                    if (usageType == DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork)
                    {
                        net.GreedyBatchedTrain(trainingData,
                            batchSize,
                            ExitEvaluatorFactory,
                            WeightLearningRateFactory,
                            HidBiasLearningRateFactory,
                            VisBiasLearningRateFactory,
                            _cancelSource.Token
                            );
                    }
                    else
                    {
                        net.GreedyBatchedSupervisedTrain(trainingData, traingLabelsCoded, batchSize,
                            ExitEvaluatorFactory,
                            WeightLearningRateFactory,
                            HidBiasLearningRateFactory,
                            VisBiasLearningRateFactory,
                            _cancelSource.Token);
                    }
                }
            }
        }


        private async void ExecuteWithSystemMemory(LayerBuilderViewModel layerBuilderViewModel,
            DataConfigViewModelBase dataConfigViewModel, int batchSize, SuspendState defaultSuspendState,
            string pathBase)
        {
            using (
                var greedyTracker =
                    new EpochErrorFileTracker<double>(Path.Combine(pathBase, "GreedyTrainError.log")))
            {
                var usageType = DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork;
                var dtType = DataConfigViewModelBase.DataTransformationTypes.NoTransform;
                int validationRecords = 0, trainingRecords = 0, testRecords = 0;
                DataReaderBase<double> testReader = null;
                DataReaderBase<double> trainingReader = null;
                DataReaderBase<double> validationReader = null;
                await Dispatcher.InvokeIfRequired(() =>
                {
                    dtType = dataConfigViewModel.DataTransformationType;
                    usageType = dataConfigViewModel.NetworkUsageType;
                    validationRecords = dataConfigViewModel.ValidationRecordCount;
                    trainingRecords = dataConfigViewModel.TrainingRecordCount;
                    testRecords = dataConfigViewModel.TestRecordCount;
                    dataConfigViewModel.GetDataReaders(out trainingReader, out validationReader, out testReader);
                });

                Func<double[,], Task<IList<BitmapSource>>> imageFactory = dtType ==
                                                                          DataConfigViewModelBase
                                                                              .DataTransformationTypes
                                                                              .Subtract128Divide127
                    ? (Func<double[,], Task<IList<BitmapSource>>>)(dd => GenerateImageSourcesPosNeg(dd))
                    : dd => GenerateImageSources(dd);

                string[] validationLabels;
                Double[,] validationLabelsCoded;
                double[,] validationData = validationReader.ReadWithLabels(validationRecords, out validationLabelsCoded,
                    out validationLabels);
                Task<IList<BitmapSource>> validationImages = imageFactory(validationData);

                IList<string[]> trainingLabels;
                IList<Double[,]> traingLabelsCoded;
                IList<double[,]> trainingData = trainingReader.ReadWithLabels(trainingRecords, batchSize,
                    out traingLabelsCoded,
                    out trainingLabels);

                var bmps = new List<BitmapSource>();
                int maxTrain = 1000;

                Task<List<BitmapSource>> tGetImages = Task.Run(async () =>
                {
                    foreach (var batch in trainingData)
                    {
                        bmps.AddRange(await imageFactory(batch));
                        if (bmps.Count >= maxTrain)
                            break;
                    }
                    return bmps;
                });

                Dispatcher.InvokeIfRequired(
                    async () =>
                        TrainingSet =
                            new ObservableCollection<BitmapSource>(await tGetImages));


                if (usageType == DataConfigViewModelBase.NetworkUsageTypes.SupervisedLabellingNetwork)
                {
                    Task<IList<BitmapSource>> validationCodes = GenerateImageSources(validationLabelsCoded);
                    Dispatcher.InvokeIfRequired(
                        async () =>
                            Reconstructions =
                                new ObservableCollection<ValidationSet>((await validationImages).Zip(
                                    await validationCodes,
                                    (source, bitmapSource) =>
                                        new ValidationSet
                                        {
                                            OriginalImageSet =
                                                new ImageSet { DataImage = source, CodeImage = bitmapSource },
                                            ReconstructedImageSet = new ImageSet()
                                        })
                                    .Zip(validationLabels,
                                        (set, s) =>
                                        {
                                            set.OriginalImageSet.Label = s;
                                            return set;
                                        })));
                }
                else
                {
                    Dispatcher.InvokeIfRequired(
                        async () =>
                            Reconstructions =
                                new ObservableCollection<ValidationSet>(
                                    (await validationImages).Select(a => new ValidationSet
                                    {
                                        OriginalImageSet = new ImageSet
                                        {
                                            DataImage = a
                                        }
                                    })));
                }

                await Dispatcher.InvokeIfRequired(() =>
                {
                    ExitEvaluatorFactory = new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000);
                    NumTrainingExamples = trainingData.Sum(a => a.GetLength(0));
                });

                GPGPU dev;
                GPGPURAND rand;
                InitCuda(out dev, out rand);
                dev.SetCurrentContext();
                using (var net = new CudaAdvancedNetwork(layerBuilderViewModel.CreateLayers(dev, rand)))
                {
                    List<double[,]> identityMatrices = IdentityMatrices(dev, net);
                    net.SetDefaultMachineState(defaultSuspendState);
                    dev.SetCurrentContext();

                    net.EpochComplete += usageType ==
                                         DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork
                        ? NetEpochUnsupervisedCompleteEventHandler(pathBase, dev, validationData,
                            identityMatrices,
                            imageFactory)
                        : NetEpochSupervisedCompleteEventHandler(pathBase, validationData, identityMatrices, dev,
                            imageFactory, validationReader);

                    net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

                    if (usageType == DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork)
                    {
                        net.GreedyBatchedTrainMem(trainingData,
                            ExitEvaluatorFactory,
                            WeightLearningRateFactory,
                            HidBiasLearningRateFactory,
                            VisBiasLearningRateFactory,
                            _cancelSource.Token
                            );
                    }
                    else
                    {
                        net.GreedyBatchedSupervisedTrainMem(trainingData, traingLabelsCoded, ExitEvaluatorFactory,
                            WeightLearningRateFactory, HidBiasLearningRateFactory, VisBiasLearningRateFactory,
                            _cancelSource.Token);
                    }
                }
            }
        }


        //private async void CsvDemo(LayerBuilderViewModel layerBuilderViewModel, int numTrainingExamples, string pathBase)
        //{
        //    GPGPU dev;
        //    GPGPURAND rand;
        //    InitCuda(out dev, out rand);
        //    dev.SetCurrentContext();
        //    IDataIO<double, string> d = new CsvData(ConfigurationManager.AppSettings["CsvDataTraining"],
        //        ConfigurationManager.AppSettings["CsvDataTest"], true, true);


        //    using (var net = new CudaAdvancedNetwork(layerBuilderViewModel.CreateLayers(dev, rand)))
        //    {
        //        net.SetDefaultMachineState(SuspendState.Active);
        //        string[] lbl;
        //        Double[,] coded;

        //        double[,] tdata = d.ReadTestData(0, 50);
        //        List<double[,]> identityMatrices = IdentityMatrices(dev, net);

        //        IList<BitmapSource> originalTestImages = await GenerateImageSources(tdata);

        //        Dispatcher.InvokeIfRequired(
        //            () =>
        //                Reconstructions =
        //                    new ObservableCollection<ValidationSet>(
        //                        originalTestImages.Select(a => new ValidationSet {DataImage = a})));

        //        dev.SetCurrentContext();

        //        net.EpochComplete += NetEpochUnsupervisedCompleteEventHandler(pathBase, dev, tdata, identityMatrices,
        //            dd => GenerateImageSources(dd));

        //        net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

        //        //batch the data in gpu memory
        //        using (
        //            var greedyTracker =
        //                new EpochErrorFileTracker<double>(Path.Combine(pathBase, "GreedyTrainError.log")))
        //        {
        //            ExitEvaluatorFactory =
        //                await
        //                    Dispatcher.InvokeIfRequired(
        //                        () => new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000));

        //            string[] lbla;
        //            Double[,] codeda;
        //            double[,] trainingData = d.ReadTrainingData(0, numTrainingExamples, out lbla, out codeda);
        //            Dispatcher.InvokeIfRequired(
        //                async () =>
        //                    TrainingSet =
        //                        new ObservableCollection<BitmapSource>(await GenerateImageSources(trainingData, 1000)));

        //            Dispatcher.InvokeIfRequired(() => NumTrainingExamples = trainingData.GetLength(0));

        //            await Dispatcher.InvokeIfRequired(() =>
        //            {
        //                WeightLearningRateFactory =
        //                    new LayerSpecificLearningRateCalculatorFactory<double>(
        //                        layerBuilderViewModel.LayerConstructionInfo.Select(
        //                            a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //                ;
        //                HidBiasLearningRateFactory =
        //                    new LayerSpecificLearningRateCalculatorFactory<double>(
        //                        layerBuilderViewModel.LayerConstructionInfo.Select(
        //                            a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //                VisBiasLearningRateFactory =
        //                    new LayerSpecificLearningRateCalculatorFactory<double>(
        //                        layerBuilderViewModel.LayerConstructionInfo.Select(
        //                            a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //            });

        //            await Dispatcher.InvokeIfRequired(() =>
        //            {
        //                var lrEditor = new ConfigureLearningRates {DataContext = this, Owner = Window.GetWindow(this)};
        //                lrEditor.ShowDialog();
        //            });

        //            //var trainingData = d.ReadTestData(0, numTrainingExamples);
        //            dev.SetCurrentContext();
        //            net.GreedyBatchedTrain(trainingData,
        //                600,
        //                ExitEvaluatorFactory,
        //                WeightLearningRateFactory,
        //                HidBiasLearningRateFactory,
        //                VisBiasLearningRateFactory,
        //                _cancelSource.Token
        //                );
        //        }

        //        //double[,] testData = d.ReadTrainingData(0, 200, out lbl, out coded);

        //        //double[,] reconstructions = net.Reconstruct(testData);

        //        //DisplayResults(pathBase, d, reconstructions, testData, lbl);

        //        //IDataIO<TElement, string> d2 = new CsvData(ConfigurationManager.AppSettings["CsvDataTest"],
        //        //    ConfigurationManager.AppSettings["CsvDataTest"], true, true);

        //        //string[] labels;
        //        //TElement[,] lcoded;
        //        //double[,] allDataToCode = d2.ReadTrainingData(0, 185945, out labels, out lcoded);
        //        //double[,] encoded = net.Encode(allDataToCode);
        //        //string[] kkey = KeyEncoder.CreateBinaryStringKeys(encoded);

        //        //using (FileStream fs = File.OpenWrite(Path.Combine(pathBase, "Encoded.csv")))
        //        //using (var tw = new StreamWriter(fs))
        //        //{
        //        //    for (int i = 0; i < allDataToCode.GetLength(0); i++)
        //        //    {
        //        //        tw.WriteLine("{0},\"{1}\"", labels[i], kkey[i]);
        //        //    }
        //        //}
        //    }
        //}

        private EventHandler<EpochEventArgs<double>> NetEpochUnsupervisedCompleteEventHandler(string pathBase, GPGPU dev,
            double[,] tdata,
            List<double[,]> identityMatrices, Func<double[,], Task<IList<BitmapSource>>> imgGenerator)
        {
            return async (a, b) =>
            {
                var nn = ((ICudaNetwork<double>)a);
                IAdvancedRbmCuda<double> m = nn.Machines[b.Layer];
                if (b.Epoch > 0 && b.Epoch % BackupFrequency == 0)
                {
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                            m.NumHiddenNeurons,
                            typeof(Double).Name, b.Epoch)));
                }


                if (b.Epoch % UpdateFrequency == 0)
                {
                    double[,] activations = GetActivations(dev, nn, tdata, b);
                    double[,] dreams = ((CudaAdvancedNetwork)nn).Daydream(1.0, 100, b.Layer);
                    double[,] recon = nn.Reconstruct(tdata, b.Layer);
                    double[,] feats = nn.Decode(identityMatrices[b.Layer], b.Layer);


                    Task.Run(() => UpdateUIProperties(pathBase, b, recon, feats, activations, dreams, imgGenerator));
                }
                else
                {
                    double[,] activations;
                    if (UpdateActivationsEveryEpoch)
                    {
                        activations = GetActivations(dev, nn, tdata, b);
                    }
                    else
                    {
                        activations = null;
                    }
                    Task.Run(() => UpdateUIProperties(pathBase, b, activations));
                }
            };
        }

        private static EventHandler<EpochEventArgs<double>> NetOnLayerTrainComplete(string pathBase)
        {
            return (a, b) =>
            {
                IAdvancedRbmCuda<double> m = ((ICudaNetwork<double>)a).Machines[b.Layer];
                m.Save(Path.Combine(pathBase,
                    string.Format("Layer_{0}_{1}x{2}_{3}_Final.dat", b.Layer, m.NumVisibleNeurons,
                        m.NumHiddenNeurons,
                        typeof(Double).Name)));
            };
        }


        //private async void FacesDemo(LayerBuilderViewModel builderViewModel, int numTrainingExamples, string pathBase)
        //{
        //    GPGPU dev;
        //    GPGPURAND rand;
        //    InitCuda(out dev, out rand);

        //    dev.SetCurrentContext();
        //    bool useLinear = builderViewModel.LayerConstructionInfo[0] is ConstructLinearHiddenLayer
        //                     || (builderViewModel.LayerConstructionInfo[0] is LoadLayerInfo
        //                         &&
        //                         ((LoadLayerInfo) (builderViewModel.LayerConstructionInfo[0])).LayerType ==
        //                         LoadLayerType.Linear);

        //    IDataIO<double, string> dataProvider =
        //        new FacesData(ConfigurationManager.AppSettings["FacesDirectory"],
        //            ConfigurationManager.AppSettings["FacesTestDirectory"],
        //            FacesData.ConversionMode.RgbToGreyPosNegReal);


        //    Func<double[,], Task<IList<BitmapSource>>> imgGenerationMethod = useLinear
        //        ? (Func<double[,], Task<IList<BitmapSource>>>) (dd => GenerateImageSourcesPosNeg(dd))
        //        : (dd => GenerateImageSources(dd));

        //    using (var net = new CudaAdvancedNetwork(builderViewModel.CreateLayers(dev, rand)))
        //    {
        //        net.SetDefaultMachineState(SuspendState.Suspended);
        //        //keep data in main memory as much as possible at the expense of more memory movement between System and GPU

        //        double[,] tdata = dataProvider.ReadTestData(numTrainingExamples, 50);
        //        DirectoryInfo di = Directory.CreateDirectory(Path.Combine(pathBase, "Original"));

        //        dev.SetCurrentContext();
        //        List<double[,]> identityMatrices = IdentityMatrices(dev, net);

        //        Task.Run(() => Dispatcher.InvokeIfRequired(async () =>
        //            Reconstructions =
        //                new ObservableCollection<ValidationSet>(
        //                    (await imgGenerationMethod(tdata)).Select(a => new ValidationSet {DataImage = a}))));

        //        dev.SetCurrentContext();

        //        net.EpochComplete += NetEpochUnsupervisedCompleteEventHandler(pathBase, dev, tdata, identityMatrices,
        //            imgGenerationMethod);
        //        net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

        //        IList<string[]> lbl;
        //        IList<double[,]> coded;

        //        IList<double[,]> training = dataProvider.ReadTrainingData(0, numTrainingExamples, 50, out lbl,
        //            out coded);

        //        Dispatcher.InvokeIfRequired(() => NumTrainingExamples = training.Sum(a => a.GetLength(0)));

        //        //await (() => NumTrainingExamples = training.Sum(a => a.GetLength(0))).InvokeIfRequired(Dispatcher);

        //        int maxtrain = 1000;

        //        var bmps = new List<BitmapSource>(maxtrain);


        //        Task<List<BitmapSource>> tgen = Task.Run(async () =>
        //        {
        //            foreach (var batch in training)
        //            {
        //                bmps.AddRange(await GenerateImageSourcesPosNeg(batch, maxtrain - bmps.Count));
        //                if (bmps.Count >= maxtrain)
        //                    break;
        //            }
        //            return bmps;
        //        });

        //        Task.Run(
        //            () =>
        //                Dispatcher.InvokeIfRequired(
        //                    async () => TrainingSet = new ObservableCollection<BitmapSource>(await tgen)));


        //        dev.SetCurrentContext();

        //        await Dispatcher.InvokeIfRequired(() =>
        //        {
        //            WeightLearningRateFactory =
        //                new LayerSpecificLearningRateCalculatorFactory<double>(
        //                    builderViewModel.LayerConstructionInfo.Select(
        //                        a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //            ;
        //            HidBiasLearningRateFactory =
        //                new LayerSpecificLearningRateCalculatorFactory<double>(
        //                    builderViewModel.LayerConstructionInfo.Select(
        //                        a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //            VisBiasLearningRateFactory =
        //                new LayerSpecificLearningRateCalculatorFactory<double>(
        //                    builderViewModel.LayerConstructionInfo.Select(
        //                        a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //        });

        //        await Dispatcher.InvokeIfRequired(() =>
        //        {
        //            var lrEditor = new ConfigureLearningRates {DataContext = this, Owner = Window.GetWindow(this)};
        //            lrEditor.ShowDialog();
        //        });

        //        using (
        //            var greedyTracker =
        //                new EpochErrorFileTracker<double>(Path.Combine(pathBase, "GreedyTrainError.log")))
        //        {
        //            ExitEvaluatorFactory =
        //                await
        //                    Dispatcher.InvokeIfRequired(
        //                        () => new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000));

        //            dev.SetCurrentContext();
        //            net.GreedyBatchedTrainMem(training,
        //                ExitEvaluatorFactory,
        //                WeightLearningRateFactory,
        //                HidBiasLearningRateFactory,
        //                VisBiasLearningRateFactory,
        //                _cancelSource.Token
        //                );
        //        }
        //    }
        //}


        //private async void KaggleDemo(LayerBuilderViewModel builderViewModel, int numTrainingExamples, string pathBase)
        //{
        //    GPGPU dev;
        //    GPGPURAND rand;
        //    InitCuda(out dev, out rand);
        //    dev.SetCurrentContext();
        //    IDataIO<double, int> dataProvider =
        //        new KaggleData(ConfigurationManager.AppSettings["KaggleTrainingData"],
        //            ConfigurationManager.AppSettings["KaggleTestData"]);


        //    using (var net = new CudaAdvancedNetwork(builderViewModel.CreateLayers(dev, rand)))
        //    {
        //        //keep data in gpu memory as much as possible
        //        net.SetDefaultMachineState(SuspendState.Active);


        //        int[] lbl;
        //        Double[,] coded;
        //        double[,] tdata = dataProvider.ReadTestData(0, 50);
        //        DirectoryInfo di = Directory.CreateDirectory(Path.Combine(pathBase, "Original"));
        //        List<double[,]> identityMatrices = IdentityMatrices(dev, net);

        //        Task.Run(() => Dispatcher.InvokeIfRequired(async () =>
        //            Reconstructions =
        //                new ObservableCollection<ValidationSet>(
        //                    (await GenerateImageSources(tdata)).Select(a => new ValidationSet {DataImage = a}))));

        //        dev.SetCurrentContext();

        //        net.EpochComplete += NetEpochSupervisedCompleteEventHandler(pathBase, tdata, identityMatrices, dev,
        //            dd => GenerateImageSources(dd));
        //        net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

        //        double[,] trainingData = dataProvider.ReadTrainingData(0, numTrainingExamples, out lbl, out coded);
        //        Task.Run(() =>
        //            Dispatcher.InvokeIfRequired(
        //                async () =>
        //                    TrainingSet =
        //                        new ObservableCollection<BitmapSource>(await GenerateImageSources(trainingData, 1000))));

        //        Dispatcher.InvokeIfRequired(() => NumTrainingExamples = trainingData.GetLength(0));

        //        using (
        //            var greedyTracker =
        //                new EpochErrorFileTracker<double>(Path.Combine(pathBase, "GreedyTrainError.log")))
        //        {
        //            ExitEvaluatorFactory =
        //                await
        //                    Dispatcher.InvokeIfRequired(
        //                        () => new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000));


        //            await Dispatcher.InvokeIfRequired(() =>
        //            {
        //                WeightLearningRateFactory =
        //                    new LayerSpecificLearningRateCalculatorFactory<double>(
        //                        builderViewModel.LayerConstructionInfo.Select(
        //                            a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //                ;
        //                HidBiasLearningRateFactory =
        //                    new LayerSpecificLearningRateCalculatorFactory<double>(
        //                        builderViewModel.LayerConstructionInfo.Select(
        //                            a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //                VisBiasLearningRateFactory =
        //                    new LayerSpecificLearningRateCalculatorFactory<double>(
        //                        builderViewModel.LayerConstructionInfo.Select(
        //                            a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
        //            });

        //            await Dispatcher.InvokeIfRequired(() =>
        //            {
        //                var lrEditor = new ConfigureLearningRates {DataContext = this, Owner = Window.GetWindow(this)};
        //                lrEditor.ShowDialog();
        //            });

        //            dev.SetCurrentContext();
        //            net.GreedyBatchedSupervisedTrain(
        //                trainingData,
        //                coded, 100,
        //                ExitEvaluatorFactory,
        //                WeightLearningRateFactory,
        //                HidBiasLearningRateFactory,
        //                VisBiasLearningRateFactory,
        //                _cancelSource.Token
        //                );
        //        }
        //    }
        //}

        private EventHandler<EpochEventArgs<double>> NetEpochSupervisedCompleteEventHandler(string pathBase,
            double[,] tdata, List<double[,]> identityMatrices, GPGPU dev,
            Func<double[,], Task<IList<BitmapSource>>> imgGenerator, DataReaderBase<double> trainingReader)
        {
            return async (a, b) =>
            {
                var nn = ((ICudaNetwork<double>)a);
                IAdvancedRbmCuda<double> m = nn.Machines[b.Layer];
                if (b.Epoch > 0 && b.Epoch % BackupFrequency == 0)
                {
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                            m.NumHiddenNeurons,
                            typeof(Double).Name, b.Epoch)));
                }

                if (b.Epoch % UpdateFrequency == 0)
                {
                    double[,] dreams;
                    double[,] recon;
                    double[,] feats;
                    double[,] activations;
                    double[,] daydreamLabelsEncoded = null;
                    double[,] encodedValidationLabels = null;
                    double[,] encodedFeatureLabels = null;
                    string[] validationLabels = null;
                    if (b.Layer == nn.Machines.Count - 1)
                    {
                        dreams = ((CudaAdvancedNetwork)nn).DaydreamWithLabels(1.0, 100, out daydreamLabelsEncoded);
                        recon = nn.ReconstructWithLabels(tdata, out encodedValidationLabels);
                        validationLabels = trainingReader.DecodeLabels(encodedValidationLabels, 1.0, 0.0);
                        feats = nn.DecodeWithLabels(identityMatrices[b.Layer], out encodedFeatureLabels);
                        activations = GetActivationsWithLabelExpansion(dev, nn, tdata);
                    }
                    else
                    {
                        dreams = ((CudaAdvancedNetwork)nn).Daydream(1.0, 100, b.Layer);
                        recon = nn.Reconstruct(tdata, b.Layer);
                        feats = nn.Decode(identityMatrices[b.Layer], b.Layer);
                        activations = GetActivations(dev, nn, tdata, b);
                    }


                    Task.Run(() => UpdateUIProperties(pathBase, b, recon, feats, activations, dreams, imgGenerator, encodedValidationLabels, validationLabels));
                }
                else
                {
                    double[,] activations;
                    if (UpdateActivationsEveryEpoch)
                    {
                        if (b.Layer == nn.Machines.Count - 1)
                        {
                            activations = GetActivationsWithLabelExpansion(dev, nn, tdata);
                        }
                        else
                        {
                            activations = GetActivations(dev, nn, tdata, b);
                        }
                    }
                    else
                    {
                        activations = null;
                    }
                    Task.Run(() => UpdateUIProperties(pathBase, b, activations));
                }
            };
        }


        private static double[,] GetActivations(GPGPU dev, ICudaNetwork<double> nn, double[,] tdata,
            EpochEventArgs<double> b)
        {
            Double[,] activations;
            using (Matrix2D<double> enc = dev.Upload(nn.Encode(tdata, b.Layer)))
            using (Matrix2D<double> act = enc.SumColumns())
            using (Matrix2D<double> sm = act.Multiply(1.0 / tdata.GetLength(0)))
            {
                activations = sm.CopyLocal();
            }
            return activations;
        }

        private static double[,] GetActivationsWithLabelExpansion(GPGPU dev, ICudaNetwork<double> nn, double[,] tdata)
        {
            Double[,] activations;
            using (Matrix2D<double> d = dev.Upload(tdata))
            using (Matrix2D<double> enc = nn.EncodeWithLabelExpansion(d))
            using (Matrix2D<double> act = enc.SumColumns())
            using (Matrix2D<double> sm = act.Multiply(1.0 / tdata.GetLength(0)))
            {
                activations = sm.CopyLocal();
            }
            return activations;
        }

        private static List<double[,]> IdentityMatrices(GPGPU dev, CudaAdvancedNetwork net)
        {
            var identityMatrices = new List<double[,]>();

            dev.SetCurrentContext();
            foreach (var advancedRbmCuda in net.Machines)
            {
                using (
                    Matrix2D<double> m = dev.AllocateNoSet<Double>(advancedRbmCuda.NumHiddenNeurons,
                        advancedRbmCuda.NumHiddenNeurons))
                {
                    m.Identity();
                    identityMatrices.Add(m.CopyLocal());
                }
            }
            return identityMatrices;
        }

        private async Task UpdateUIProperties(string pathBase, EpochEventArgs<double> epochEventArgs,
            Double[,] activations)
        {
            Epoch = epochEventArgs.Epoch;
            Error = epochEventArgs.Error;
            Layer = epochEventArgs.Layer;
            Delta = epochEventArgs.Delta;
            Elapsed = epochEventArgs.Elapsed;
            LearningRate = epochEventArgs.LearningRate;

            if (activations != null)
            {
                IList<BitmapSource> actiim = await GenerateImageSources(activations);

                await Dispatcher.InvokeIfRequired(() => { ActivationFrequency = actiim[0]; });
            }
        }

        private async void UpdateUIProperties(string pathBase, EpochEventArgs<double> b, double[,] recon,
            double[,] feats,
            double[,] activations, double[,] dreams, Func<double[,], Task<IList<BitmapSource>>> imgGenerator,
            double[,] validationLabelsEncoded = null, string[] validationLabels = null)
        {
            DisplayedEpoch = b.Epoch;
            Task t = UpdateUIProperties(pathBase, b, activations);

            Task t1 =
                Task.Run(
                    async () =>
                        UpdateImageResult(Reconstructions, await imgGenerator(recon),
                            validationLabels == null ? null : await GenerateImageSources(validationLabelsEncoded),
                            validationLabels));
            Task<Task> t2 = Task.Run(
                () =>
                    Dispatcher.InvokeIfRequired(
                        async () => { DayDreams = new ObservableCollection<BitmapSource>(await imgGenerator(dreams)); }));
            Task<Task> t3 = Task.Run(
                () =>
                    Dispatcher.InvokeIfRequired(
                        async () =>
                        {
                            Features = new ObservableCollection<BitmapSource>(await imgGenerator(feats));
                            SelectFeature(SelectedFeatureIndex);
                        }));

            await Task.Run(() => Task.WaitAll(t, t1, t2, t3));
        }

        private async void UpdateImageResult(ObservableCollection<ValidationSet> set, IList<BitmapSource> reconIm,
            IList<BitmapSource> labelsEncoded, string[] labels)
        {
            await Dispatcher.InvokeIfRequired(() =>
            {
                for (int i = 0; i < reconIm.Count; i++)
                {
                    set[i].ReconstructedImageSet = new ImageSet
                    {
                        DataImage = reconIm[i],
                        Label = labels == null ? null : labels[i],
                        CodeImage = labelsEncoded == null ? null : labelsEncoded[i]
                    };
                }
            });
        }


        private async void DisplayResults<TLabel>(string pathBase, IDataIO<double, TLabel> dataProvider,
            Double[,] reconstructions, Double[,] referenceData, TLabel[] labels,
            Double[,] referenceCode = null, Double[,] computedCode = null)
        {
            dataProvider.PrintToConsole(reconstructions, referenceData, labels, referenceCode,
                computedLabels: computedCode);
            IList<BitmapSource> finalTest = await GenerateImageSources(referenceData);
            IList<BitmapSource> finalRecon = await GenerateImageSources(reconstructions);
            Reconstructions =
                new ObservableCollection<ValidationSet>(finalTest.Zip(finalRecon,
                    (a, b) =>
                        new ValidationSet
                        {
                            OriginalImageSet = new ImageSet { DataImage = a },
                            ReconstructedImageSet = new ImageSet { DataImage = b }
                        }));
        }

        private async Task<IList<BitmapSource>> GenerateImageSources(Double[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte)(b * 255f), maxResults);
        }

        private async Task<IList<BitmapSource>> GenerateImageSources(Double[,] data,
            Func<double, byte> converter, int maxResults)
        {
            int num = Math.Min(data.GetLength(0), maxResults);
            var bmps = new Bitmap[num];

            await Task.Run(() =>
            {
                for (int a = 0; a < num; a++)
                {
                    int stride;
                    bmps[a] = ImageUtils.GenerateBitmap(data, a, converter, out stride);
                }
            });

            var images = new BitmapSource[data.GetLength(0)];

            await Dispatcher.InvokeIfRequired(
                () =>
                {
                    for (int a = 0; a < num; a++)
                    {
                        IntPtr h = bmps[a].GetHbitmap();
                        try
                        {
                            images[a] =
                                Imaging.CreateBitmapSourceFromHBitmap(h, IntPtr.Zero, Int32Rect.Empty,
                                    BitmapSizeOptions.FromEmptyOptions());
                            images[a].Freeze();
                        }
                        finally
                        {
                            DeleteObject(h);
                        }
                    }
                });
            foreach (Bitmap bitmap in bmps)
            {
                bitmap.Dispose();
            }
            return images;
        }

        [DllImport("gdi32", EntryPoint = "DeleteObject")]
        private static extern int DeleteObject(IntPtr o);


        private async Task<IList<BitmapSource>> GenerateImageSourcesInt(Double[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte)(b), maxResults);
        }

        private async Task<IList<BitmapSource>> GenerateImageSourcesPosNeg(
            Double[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte)((b * 127.0) + 128.0), maxResults);
        }


        private static void InitCuda(out GPGPU dev, out GPGPURAND rand)
        {
            CudafyHost.ClearAllDeviceMemories();
            CudafyHost.ClearDevices();


            dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            GPGPUProperties props = dev.GetDeviceProperties(false);
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            eArchitecture arch = dev.GetArchitecture();


            ePlatform plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            string kernelPath = Path.Combine(Environment.CurrentDirectory,
                string.Format("CudaKernels_{0}.kernel", plat));

            CudafyModule mod;
            if (File.Exists(kernelPath))
            {
                Console.WriteLine("Loading kernels  from {0}", kernelPath);
                mod = CudafyModule.Deserialize(kernelPath);
            }
            else
            {
                Console.WriteLine("Compiling cuda kernels");
                mod = CudafyTranslator.Cudafy(
                    plat,
                    arch,
                    typeof(ActivationFunctionsCuda),
                    typeof(Matrix2DCuda)
                    );
                Console.WriteLine("Saving kernels to {0}", kernelPath);
                mod.Serialize(kernelPath);
            }

            ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount,
                props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            rand = GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

            rand.SetPseudoRandomGeneratorSeed((ulong)DateTime.Now.Ticks);
            rand.GenerateSeeds();

            Console.WriteLine("Loading Module");
            dev.LoadModule(mod);
        }


        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);

            if (e.Property == ErrorProperty)
            {
                ErrorLabelBrush = (double)e.OldValue > (double)e.NewValue
                    ? new SolidColorBrush(Colors.Blue)
                    : new SolidColorBrush(Colors.Red);
            }

            if (e.Property == DeltaProperty)
            {
                if ((double)e.NewValue > 0)
                {
                    DeltaLabelBrush = (double)e.OldValue > (double)e.NewValue
                        ? new SolidColorBrush(Colors.Orange)
                        : new SolidColorBrush(Colors.Blue);
                }
                else
                {
                    DeltaLabelBrush = (double)e.OldValue < (double)e.NewValue
                        ? new SolidColorBrush(Colors.Orange)
                        : new SolidColorBrush(Colors.Red);
                }
            }
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                _cancelSource.Cancel();
            }
        }
    }
}