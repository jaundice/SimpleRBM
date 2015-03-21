using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
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
#else
using TElement = System.Double;
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
            typeof(TElement), typeof(BelieveViewModel), new PropertyMetadata(default(TElement)));

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

        public static readonly DependencyProperty ErrorProperty = DependencyProperty.Register("Error", typeof(TElement),
            typeof(BelieveViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty DeltaProperty = DependencyProperty.Register("Delta", typeof(TElement),
            typeof(BelieveViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty NumTrainingExamplesProperty =
            DependencyProperty.Register("NumTrainingExamples", typeof(int), typeof(BelieveViewModel),
                new PropertyMetadata(1000));

        public static readonly DependencyProperty DayDreamsProperty = DependencyProperty.Register("DayDreams",
            typeof(ObservableCollection<BitmapSource>), typeof(BelieveViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty TrainingSetProperty = DependencyProperty.Register("TrainingSet",
            typeof(ObservableCollection<ImageSet>), typeof(BelieveViewModel),
            new PropertyMetadata(default(ObservableCollection<ImageSet>)));

        public static readonly DependencyProperty ElapsedProperty = DependencyProperty.Register("Elapsed",
            typeof(TimeSpan), typeof(BelieveViewModel), new PropertyMetadata(default(TimeSpan)));

        public static readonly DependencyProperty ExitEvaluatorFactoryProperty =
            DependencyProperty.Register("ExitEvaluatorFactory", typeof(InteractiveExitEvaluatorFactory<TElement>),
                typeof(BelieveViewModel), new PropertyMetadata(default(InteractiveExitEvaluatorFactory<TElement>)));

        public static readonly DependencyProperty DisplayedEpochProperty = DependencyProperty.Register(
            "DisplayedEpoch", typeof(int), typeof(BelieveViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty WeightLearningRateFactoryProperty =
            DependencyProperty.Register("WeightLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<TElement>), typeof(BelieveViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<TElement>)));

        public static readonly DependencyProperty DisplayFeatureCommandProperty =
            DependencyProperty.Register("DisplayFeatureCommand", typeof(ICommand), typeof(BelieveViewModel),
                new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty SelectedFeatureProperty =
            DependencyProperty.Register("SelectedFeature", typeof(BitmapSource), typeof(BelieveViewModel),
                new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty VisBiasLearningRateFactoryProperty =
            DependencyProperty.Register("VisBiasLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<TElement>), typeof(BelieveViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<TElement>)));

        public static readonly DependencyProperty HidBiasLearningRateFactoryProperty =
            DependencyProperty.Register("HidBiasLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<TElement>), typeof(BelieveViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<TElement>)));

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
#if USEFLOAT
                float rate = Convert.ToSingle(a);
#else
                double rate = Convert.ToDouble(a);
#endif
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

        public TElement LearningRate
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement)GetValue(LearningRateProperty)).Result; }
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

        public TElement Error
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement)GetValue(ErrorProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ErrorProperty, value)).Wait(); }
        }

        public TElement Delta
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement)GetValue(DeltaProperty)).Result; }
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

        public ObservableCollection<ImageSet> TrainingSet
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (ObservableCollection<ImageSet>)GetValue(TrainingSetProperty))
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

        public InteractiveExitEvaluatorFactory<TElement> ExitEvaluatorFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () => (InteractiveExitEvaluatorFactory<TElement>)GetValue(ExitEvaluatorFactoryProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ExitEvaluatorFactoryProperty, value)).Wait(); }
        }

        public int DisplayedEpoch
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(DisplayedEpochProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DisplayedEpochProperty, value)).Wait(); }
        }

        public LayerSpecificLearningRateCalculatorFactory<TElement> WeightLearningRateFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () =>
                            (LayerSpecificLearningRateCalculatorFactory<TElement>)
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

        public LayerSpecificLearningRateCalculatorFactory<TElement> VisBiasLearningRateFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () =>
                            (LayerSpecificLearningRateCalculatorFactory<TElement>)
                                GetValue(VisBiasLearningRateFactoryProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(VisBiasLearningRateFactoryProperty, value)).Wait(); }
        }

        public LayerSpecificLearningRateCalculatorFactory<TElement> HidBiasLearningRateFactory
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () =>
                            (LayerSpecificLearningRateCalculatorFactory<TElement>)
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
                new LayerSpecificLearningRateCalculatorFactory<TElement>(
                    builder.LayerConstructionInfo.Select(
                        a => new InteractiveLearningRateCalculatorFactory<TElement>((TElement)3E-05)));
            HidBiasLearningRateFactory =
                new LayerSpecificLearningRateCalculatorFactory<TElement>(
                    builder.LayerConstructionInfo.Select(
                        a => new InteractiveLearningRateCalculatorFactory<TElement>((TElement)3E-05)));
            VisBiasLearningRateFactory =
                new LayerSpecificLearningRateCalculatorFactory<TElement>(
                    builder.LayerConstructionInfo.Select(
                        a => new InteractiveLearningRateCalculatorFactory<TElement>((TElement)3E-05)));
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
                WeightInitializationStDev = (TElement)0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500,
                NumHiddenNeurons = 500,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = (TElement)0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500 + labelWidth,
                NumHiddenNeurons = 2000,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = (TElement)0.01
            });
        }

        private void ConfigureDefaultDataLayers(LayerBuilderViewModel builderViewModel, int dataWidth, int labelWidth)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = dataWidth,
                NumHiddenNeurons = 500,
                ConvertActivationsToStates = false,
                WeightInitializationStDev = (TElement)0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500,
                NumHiddenNeurons = 500,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = (TElement)0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 500 + labelWidth,
                NumHiddenNeurons = 50,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = (TElement)0.01
            });
        }

        private void ConfigureDefaultFacesLayers(LayerBuilderViewModel builderViewModel, int dataWidth, int labelWidth)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructLinearHiddenLayer
            {
                NumVisibleNeurons = dataWidth,
                NumHiddenNeurons = 2000,
                WeightInitializationStDev = (TElement)0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructLinearHiddenLayer
            {
                NumVisibleNeurons = 2000,
                NumHiddenNeurons = 4000,
                WeightInitializationStDev = (TElement)0.01
            });
            builderViewModel.LayerConstructionInfo.Add(new ConstructLinearHiddenLayer
            {
                NumVisibleNeurons = 4000 + labelWidth,
                NumHiddenNeurons = 4000,
                WeightInitializationStDev = (TElement)0.01
            });
        }


        private async void ExecuteWithGPUMemory(LayerBuilderViewModel layerBuilderViewModel,
            DataConfigViewModelBase dataConfigViewModel, int batchSize, SuspendState defaultSuspendState,
            string pathBase)
        {
            using (
                var greedyTracker =
                    new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
            {
                int validationRecords = 0, trainingRecords = 0, testRecords = 0;
                var usageType = DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork;
                var dtType = DataConfigViewModelBase.DataTransformationTypes.NoTransform;
                int startTrainingLayer = 0;
                DataReaderBase<TElement> testReader = null;
                DataReaderBase<TElement> trainingReader = null;
                DataReaderBase<TElement> validationReader = null;
                await Dispatcher.InvokeIfRequired(() =>
                {
                    startTrainingLayer = layerBuilderViewModel.StartTrainLayerIndex;
                    dtType = dataConfigViewModel.DataTransformationType;
                    usageType = dataConfigViewModel.NetworkUsageType;
                    validationRecords = dataConfigViewModel.ValidationRecordCount;
                    trainingRecords = dataConfigViewModel.TrainingRecordCount;
                    testRecords = dataConfigViewModel.TestRecordCount;
                    dataConfigViewModel.GetDataReaders(out trainingReader, out validationReader, out testReader);
                });

                Func<TElement[,], Task<IList<BitmapSource>>> imageFactory = dtType ==
                                                                          DataConfigViewModelBase
                                                                              .DataTransformationTypes
                                                                              .Subtract128Divide127
                    ? (Func<TElement[,], Task<IList<BitmapSource>>>)(dd => GenerateImageSourcesPosNeg(dd))
                    : dd => GenerateImageSources(dd);


                string[] validationLabels;
                TElement[,] validationLabelsCoded;
                TElement[,] validationData = validationReader.ReadWithLabels(validationRecords, out validationLabelsCoded,
                    out validationLabels);
                Task<IList<BitmapSource>> validationImages = imageFactory(validationData);


                string[] trainingLabels;
                TElement[,] trainingLabelsCoded;
                TElement[,] trainingData = trainingReader.ReadWithLabels(trainingRecords, out trainingLabelsCoded,
                    out trainingLabels);


                if (usageType == DataConfigViewModelBase.NetworkUsageTypes.SupervisedLabellingNetwork)
                {
                    await Dispatcher.InvokeIfRequired(
                        async () =>
                            TrainingSet =
                                new ObservableCollection<ImageSet>((await imageFactory(trainingData)).Zip(
                                    await GenerateImageSources(trainingLabelsCoded), (a, b) => new ImageSet
                                    {
                                        DataImage = a,
                                        CodeImage = b
                                    }).Zip(trainingLabels, (a, b) =>
                                    {
                                        a.Label = b;
                                        return a;
                                    })));
                }
                else
                {
                   await Dispatcher.InvokeIfRequired(
                        async () =>
                            TrainingSet =
                                new ObservableCollection<ImageSet>(
                                    (await imageFactory(trainingData)).Select(a => new ImageSet { DataImage = a })));
                }


                if (usageType == DataConfigViewModelBase.NetworkUsageTypes.SupervisedLabellingNetwork)
                {
                    Task<IList<BitmapSource>> validationCodes = GenerateImageSources(validationLabelsCoded);
                   await Dispatcher.InvokeIfRequired(
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
                    await Dispatcher.InvokeIfRequired(
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
                    ExitEvaluatorFactory = new InteractiveExitEvaluatorFactory<TElement>(greedyTracker, (TElement)0.5, 5000);
                    NumTrainingExamples = trainingData.GetLength(0);
                });

                GPGPU dev;
                GPGPURAND rand;
                InitCuda(out dev, out rand);
                dev.SetCurrentContext();
                using (var net = new CudaAdvancedNetwork(layerBuilderViewModel.CreateLayers(dev, rand)))
                {
                    List<TElement[,]> identityMatrices = IdentityMatrices(dev, net);
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
                            _cancelSource.Token,
                            startTrainingLayer
                            );

                        var codeBatches = testReader.Read(testRecords, batchSize).Select(a => net.Encode(a));
                        var codeStringBatches = codeBatches.Select(GenerateCodeStrings);
                        foreach (var batch in codeStringBatches)
                        {
                            File.AppendAllLines(Path.Combine(pathBase, "ComputedCodes.csv"), batch);
                        }

                    }
                    else
                    {
                        net.GreedyBatchedSupervisedTrain(trainingData, trainingLabelsCoded, batchSize,
                            ExitEvaluatorFactory,
                            WeightLearningRateFactory,
                            HidBiasLearningRateFactory,
                            VisBiasLearningRateFactory,
                            _cancelSource.Token,
                            startTrainingLayer);

                        var labelCodeBatches = testReader.Read(testRecords, batchSize).Select(a => net.LabelData(a));

                        var labelBatches = labelCodeBatches.Select(a => testReader.DecodeLabels(a, (TElement)1.0, (TElement)0.0));

                        foreach (var batch in labelBatches)
                        {
                            File.AppendAllLines(Path.Combine(pathBase, "ComputedLabels.csv"), batch);

                        }


                    }
                }
            }
        }

        private string[] GenerateCodeStrings(TElement[,] codes)
        {
            var res = new string[codes.GetLength(0)];
            Parallel.For(0, res.Length, i =>
            {
                StringBuilder sb = new StringBuilder();
                for (var j = 0; j < codes.GetLength(1); j++)
                {
                    sb.Append(codes[i, j] < 0.5 ? "0" : "1");
                }
                res[i] = sb.ToString();
            });

            return res;
        }


        private async void ExecuteWithSystemMemory(LayerBuilderViewModel layerBuilderViewModel,
            DataConfigViewModelBase dataConfigViewModel, int batchSize, SuspendState defaultSuspendState,
            string pathBase)
        {
            using (
                var greedyTracker =
                    new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
            {
                var usageType = DataConfigViewModelBase.NetworkUsageTypes.UnsupervisedCodingNetwork;
                var dtType = DataConfigViewModelBase.DataTransformationTypes.NoTransform;
                int validationRecords = 0, trainingRecords = 0, testRecords = 0, startTrainFrom = 0;
                DataReaderBase<TElement> testReader = null;
                DataReaderBase<TElement> trainingReader = null;
                DataReaderBase<TElement> validationReader = null;
                await Dispatcher.InvokeIfRequired(() =>
                {
                    dtType = dataConfigViewModel.DataTransformationType;
                    usageType = dataConfigViewModel.NetworkUsageType;
                    validationRecords = dataConfigViewModel.ValidationRecordCount;
                    trainingRecords = dataConfigViewModel.TrainingRecordCount;
                    testRecords = dataConfigViewModel.TestRecordCount;
                    startTrainFrom = layerBuilderViewModel.StartTrainLayerIndex;
                    dataConfigViewModel.GetDataReaders(out trainingReader, out validationReader, out testReader);
                });

                Func<TElement[,], Task<IList<BitmapSource>>> imageFactory = dtType ==
                                                                          DataConfigViewModelBase
                                                                              .DataTransformationTypes
                                                                              .Subtract128Divide127
                    ? (Func<TElement[,], Task<IList<BitmapSource>>>)(dd => GenerateImageSourcesPosNeg(dd))
                    : dd => GenerateImageSources(dd);

                string[] validationLabels;
                TElement[,] validationLabelsCoded;
                TElement[,] validationData = validationReader.ReadWithLabels(validationRecords, out validationLabelsCoded,
                    out validationLabels);
                Task<IList<BitmapSource>> validationImages = imageFactory(validationData);

                IList<string[]> trainingLabels;
                IList<TElement[,]> trainingLabelsCoded;
                IList<TElement[,]> trainingData = trainingReader.ReadWithLabels(trainingRecords, batchSize,
                    out trainingLabelsCoded,
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



                if (usageType == DataConfigViewModelBase.NetworkUsageTypes.SupervisedLabellingNetwork)
                {
                    Task<List<BitmapSource>> tGetCodedLabels = Task.Run(async () =>
                    {
                        foreach (var batch in trainingLabelsCoded)
                        {
                            bmps.AddRange(await GenerateImageSources(batch));
                            if (bmps.Count >= maxTrain)
                                break;
                        }
                        return bmps;
                    });

                   await Dispatcher.InvokeIfRequired(
                        async () =>
                            TrainingSet =
                                new ObservableCollection<ImageSet>((await tGetImages).Zip(
                                   await tGetCodedLabels, (a, b) => new ImageSet
                                    {
                                        DataImage = a,
                                        CodeImage = b
                                    }).Zip(trainingLabels.SelectMany(c => c), (a, b) =>
                                    {
                                        a.Label = b;
                                        return a;
                                    })));
                }
                else
                {
                   await Dispatcher.InvokeIfRequired(
                        async () =>
                            TrainingSet =
                                new ObservableCollection<ImageSet>(
                                    (await tGetImages).Select(a => new ImageSet { DataImage = a })));
                }

                if (usageType == DataConfigViewModelBase.NetworkUsageTypes.SupervisedLabellingNetwork)
                {
                    Task<IList<BitmapSource>> validationCodes = GenerateImageSources(validationLabelsCoded);
                    await Dispatcher.InvokeIfRequired(
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
                    await Dispatcher.InvokeIfRequired(
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
                    ExitEvaluatorFactory = new InteractiveExitEvaluatorFactory<TElement>(greedyTracker, (TElement)0.5, 5000);
                    NumTrainingExamples = trainingData.Sum(a => a.GetLength(0));
                });

                GPGPU dev;
                GPGPURAND rand;
                InitCuda(out dev, out rand);
                dev.SetCurrentContext();
                using (var net = new CudaAdvancedNetwork(layerBuilderViewModel.CreateLayers(dev, rand)))
                {
                    List<TElement[,]> identityMatrices = IdentityMatrices(dev, net);
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
                            _cancelSource.Token,
                            startTrainFrom
                            );

                        var codeBatches = testReader.Read(testRecords, batchSize).Select(a => net.Encode(a));
                        var codeStringBatches = codeBatches.Select(GenerateCodeStrings);
                        foreach (var batch in codeStringBatches)
                        {
                            File.AppendAllLines(Path.Combine(pathBase, "ComputedCodes.csv"), batch);
                        }
                    }
                    else
                    {
                        net.GreedyBatchedSupervisedTrainMem(trainingData, trainingLabelsCoded, ExitEvaluatorFactory,
                            WeightLearningRateFactory, HidBiasLearningRateFactory, VisBiasLearningRateFactory,
                            _cancelSource.Token, startTrainFrom);

                        var labelCodeBatches = testReader.Read(testRecords, batchSize).Select(a => net.LabelData(a));

                        var labelBatches = labelCodeBatches.Select(a => testReader.DecodeLabels(a, (TElement)1.0, (TElement)0.0));

                        foreach (var batch in labelBatches)
                        {
                            File.AppendAllLines(Path.Combine(pathBase, "ComputedLabels.csv"), batch);

                        }
                    }
                }
            }
        }


        private EventHandler<EpochEventArgs<TElement>> NetEpochUnsupervisedCompleteEventHandler(string pathBase, GPGPU dev,
            TElement[,] tdata,
            List<TElement[,]> identityMatrices, Func<TElement[,], Task<IList<BitmapSource>>> imgGenerator)
        {
            return async (a, b) =>
            {
                var nn = ((ICudaNetwork<TElement>)a);
                IAdvancedRbmCuda<TElement> m = nn.Machines[b.Layer];
                if (b.Epoch > 0 && b.Epoch % BackupFrequency == 0)
                {
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                            m.NumHiddenNeurons,
                            typeof(TElement).Name, b.Epoch)));
                }


                if (b.Epoch % UpdateFrequency == 0)
                {
                    string[] computedCodes = null;
                    TElement[,] activations = GetActivations(dev, nn, tdata, b);
                    TElement[,] dreams = ((CudaAdvancedNetwork)nn).Daydream((TElement)1.0, 100, b.Layer);
                    TElement[,] recon = nn.Reconstruct(tdata, b.Layer);
                    TElement[,] feats = nn.Decode(identityMatrices[b.Layer], b.Layer);
                    TElement[,] codes = null;
                    if (b.Layer == nn.Machines.Count - 1)
                    {
                        codes = nn.Encode(tdata);
                        computedCodes = GenerateCodeStrings(codes);
                    }
                    await Task.Run(() => UpdateUIProperties(pathBase, b, recon, feats, activations, dreams, imgGenerator, validationLabels: computedCodes, validationLabelsEncoded: codes));
                }
                else
                {
                    TElement[,] activations;
                    if (UpdateActivationsEveryEpoch)
                    {
                        activations = GetActivations(dev, nn, tdata, b);
                    }
                    else
                    {
                        activations = null;
                    }
                    await Task.Run(() => UpdateUIProperties(pathBase, b, activations));
                }
            };
        }

        private static EventHandler<EpochEventArgs<TElement>> NetOnLayerTrainComplete(string pathBase)
        {
            return (a, b) =>
            {
                IAdvancedRbmCuda<TElement> m = ((ICudaNetwork<TElement>)a).Machines[b.Layer];
                m.Save(Path.Combine(pathBase,
                    string.Format("Layer_{0}_{1}x{2}_{3}_Final.dat", b.Layer, m.NumVisibleNeurons,
                        m.NumHiddenNeurons,
                        typeof(TElement).Name)));
            };
        }


        private EventHandler<EpochEventArgs<TElement>> NetEpochSupervisedCompleteEventHandler(string pathBase,
            TElement[,] tdata, List<TElement[,]> identityMatrices, GPGPU dev,
            Func<TElement[,], Task<IList<BitmapSource>>> imgGenerator, DataReaderBase<TElement> trainingReader)
        {
            return async (a, b) =>
            {
                var nn = ((ICudaNetwork<TElement>)a);
                IAdvancedRbmCuda<TElement> m = nn.Machines[b.Layer];
                if (b.Epoch > 0 && b.Epoch % BackupFrequency == 0)
                {
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                            m.NumHiddenNeurons,
                            typeof(TElement).Name, b.Epoch)));
                }

                if (b.Epoch % UpdateFrequency == 0)
                {
                    TElement[,] dreams;
                    TElement[,] recon;
                    TElement[,] feats;
                    TElement[,] activations;
                    TElement[,] daydreamLabelsEncoded = null;
                    TElement[,] encodedValidationLabels = null;
                    TElement[,] encodedFeatureLabels = null;
                    string[] validationLabels = null;
                    if (b.Layer == nn.Machines.Count - 1)
                    {
                        dreams = ((CudaAdvancedNetwork)nn).DaydreamWithLabels((TElement)1.0, 100, out daydreamLabelsEncoded);
                        recon = nn.ReconstructWithLabels(tdata, out encodedValidationLabels);
                        validationLabels = trainingReader.DecodeLabels(encodedValidationLabels, (TElement)1.0, (TElement)0.0);
                        feats = nn.DecodeWithLabels(identityMatrices[b.Layer], out encodedFeatureLabels);
                        activations = GetActivationsWithLabelExpansion(dev, nn, tdata);
                    }
                    else
                    {
                        dreams = ((CudaAdvancedNetwork)nn).Daydream((TElement)1.0, 100, b.Layer);
                        recon = nn.Reconstruct(tdata, b.Layer);
                        feats = nn.Decode(identityMatrices[b.Layer], b.Layer);
                        activations = GetActivations(dev, nn, tdata, b);
                    }


                    await Task.Run(
                         () =>
                             UpdateUIProperties(pathBase, b, recon, feats, activations, dreams, imgGenerator,
                                 encodedValidationLabels, validationLabels));
                }
                else
                {
                    TElement[,] activations;
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
                    await Task.Run(() => UpdateUIProperties(pathBase, b, activations));
                }
            };
        }


        private static TElement[,] GetActivations(GPGPU dev, ICudaNetwork<TElement> nn, TElement[,] tdata,
            EpochEventArgs<TElement> b)
        {
            TElement[,] activations;
            using (Matrix2D<TElement> enc = dev.Upload(nn.Encode(tdata, b.Layer)))
            using (Matrix2D<TElement> act = enc.SumColumns())
            using (Matrix2D<TElement> sm = act.Multiply((TElement)1.0 / tdata.GetLength(0)))
            {
                activations = sm.CopyLocal();
            }
            return activations;
        }

        private static TElement[,] GetActivationsWithLabelExpansion(GPGPU dev, ICudaNetwork<TElement> nn, TElement[,] tdata)
        {
            TElement[,] activations;
            using (Matrix2D<TElement> d = dev.Upload(tdata))
            using (Matrix2D<TElement> enc = nn.EncodeWithLabelExpansion(d))
            using (Matrix2D<TElement> act = enc.SumColumns())
            using (Matrix2D<TElement> sm = act.Multiply((TElement)1.0 / tdata.GetLength(0)))
            {
                activations = sm.CopyLocal();
            }
            return activations;
        }

        private static List<TElement[,]> IdentityMatrices(GPGPU dev, CudaAdvancedNetwork net)
        {
            var identityMatrices = new List<TElement[,]>();

            dev.SetCurrentContext();
            foreach (var advancedRbmCuda in net.Machines)
            {
                using (
                    Matrix2D<TElement> m = dev.AllocateNoSet<TElement>(advancedRbmCuda.NumHiddenNeurons,
                        advancedRbmCuda.NumHiddenNeurons))
                {
                    m.Identity();
                    identityMatrices.Add(m.CopyLocal());
                }
            }
            return identityMatrices;
        }

        private async Task UpdateUIProperties(string pathBase, EpochEventArgs<TElement> epochEventArgs,
            TElement[,] activations)
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

        private async void UpdateUIProperties(string pathBase, EpochEventArgs<TElement> b, TElement[,] recon,
            TElement[,] feats,
            TElement[,] activations, TElement[,] dreams, Func<TElement[,], Task<IList<BitmapSource>>> imgGenerator,
            TElement[,] validationLabelsEncoded = null, string[] validationLabels = null)
        {
            DisplayedEpoch = b.Epoch;
            Task t = UpdateUIProperties(pathBase, b, activations);

            var tRecon = Task.Run(async () => await imgGenerator(recon));
            var tValid = Task.Run(async () => validationLabels == null ? null : await GenerateImageSources(validationLabelsEncoded));
            var tDreams = Task.Run(async () => await imgGenerator(dreams));
            var tFeats = Task.Run(async () => await imgGenerator(feats));
            Task.WaitAll(tRecon, tValid, tDreams, tFeats);

            Task t1 =
                Task.Run(
                    async () =>
                        UpdateImageResult(Reconstructions, await tRecon,
                            await tValid,
                            validationLabels));
            Task<Task> t2 = Task.Run(
                () =>
                    Dispatcher.InvokeIfRequired(
                        async () => { DayDreams = new ObservableCollection<BitmapSource>(await tDreams); }));
            Task<Task> t3 = Task.Run(
                () =>
                    Dispatcher.InvokeIfRequired(
                        async () =>
                        {
                            Features = new ObservableCollection<BitmapSource>(await tFeats);
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


        private async void DisplayResults<TLabel>(string pathBase, IDataIO<TElement, TLabel> dataProvider,
            TElement[,] reconstructions, TElement[,] referenceData, TLabel[] labels,
            TElement[,] referenceCode = null, TElement[,] computedCode = null)
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

        private async Task<IList<BitmapSource>> GenerateImageSources(TElement[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte)(b * 255f), maxResults);
        }

        private async Task<IList<BitmapSource>> GenerateImageSources(TElement[,] data,
            Func<TElement, byte> converter, int maxResults)
        {
            int num = Math.Min(data.GetLength(0), maxResults);
            var bmps = new Bitmap[num];

            await Task.Run(() =>
            {
                //for (int a = 0; a < num; a++)
                Parallel.For(0, num, a =>
                {
                    int stride;
                    bmps[a] = ImageUtils.GenerateBitmap(data, a, converter, out stride);
                });
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


        private async Task<IList<BitmapSource>> GenerateImageSourcesInt(TElement[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte)(b), maxResults);
        }

        private async Task<IList<BitmapSource>> GenerateImageSourcesPosNeg(
            TElement[,] data, int maxResults = int.MaxValue)
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
                ErrorLabelBrush = (TElement)e.OldValue > (TElement)e.NewValue
                    ? new SolidColorBrush(Colors.Blue)
                    : new SolidColorBrush(Colors.Red);
            }

            if (e.Property == DeltaProperty)
            {
                if ((TElement)e.NewValue > 0)
                {
                    DeltaLabelBrush = (TElement)e.OldValue > (TElement)e.NewValue
                        ? new SolidColorBrush(Colors.Orange)
                        : new SolidColorBrush(Colors.Blue);
                }
                else
                {
                    DeltaLabelBrush = (TElement)e.OldValue < (TElement)e.NewValue
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