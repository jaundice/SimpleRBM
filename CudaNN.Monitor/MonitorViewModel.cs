using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Configuration;
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
using Mono.CSharp;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Cuda;
using SimpleRBM.Demo;
using SimpleRBM.Demo.IO;
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

namespace CudaNN.Monitor
{
    public class MonitorViewModel : DependencyObject, IDisposable
    {
        public static readonly DependencyProperty DemoModeProperty = DependencyProperty.Register("DemoMode",
            typeof(string), typeof(MonitorViewModel), new PropertyMetadata(default(string)));

        public static readonly DependencyProperty LayerProperty = DependencyProperty.Register("Layer", typeof(int),
            typeof(MonitorViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty SelectedFeatureIndexProperty =
            DependencyProperty.Register("SelectedFeatureIndex", typeof(int),
                typeof(MonitorViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty EpochProperty = DependencyProperty.Register("Epoch", typeof(int),
            typeof(MonitorViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty LearningRateProperty = DependencyProperty.Register("LearningRate",
            typeof(TElement), typeof(MonitorViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty UpdateFrequencyProperty =
            DependencyProperty.Register("UpdateFrequency", typeof(int), typeof(MonitorViewModel),
                new PropertyMetadata(200));

        public static readonly DependencyProperty ActivationFrequencyProperty =
            DependencyProperty.Register("ActivationFrequency", typeof(BitmapSource), typeof(MonitorViewModel),
                new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty ReconstructionsProperty =
            DependencyProperty.Register("Reconstructions", typeof(ObservableCollection<ImagePair>),
                typeof(MonitorViewModel), new PropertyMetadata(default(ObservableCollection<ImagePair>)));

        public static readonly DependencyProperty FeaturesProperty = DependencyProperty.Register("Features",
            typeof(ObservableCollection<BitmapSource>), typeof(MonitorViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty RunAppMethodProperty = DependencyProperty.Register("RunAppMethod",
            typeof(ICommand), typeof(MonitorViewModel), new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty RunBindingProperty = DependencyProperty.Register("RunBinding",
            typeof(CommandBinding), typeof(MonitorViewModel), new PropertyMetadata(default(CommandBinding)));

        public static readonly DependencyProperty RunCommandProperty = DependencyProperty.Register("RunCommand",
            typeof(ICommand), typeof(MonitorViewModel), new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty BackupFrequencyProperty =
            DependencyProperty.Register("BackupFrequency", typeof(int), typeof(MonitorViewModel),
                new PropertyMetadata(1000));

        public static readonly DependencyProperty ErrorProperty = DependencyProperty.Register("Error", typeof(TElement),
            typeof(MonitorViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty DeltaProperty = DependencyProperty.Register("Delta", typeof(TElement),
            typeof(MonitorViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty NumTrainingExamplesProperty =
            DependencyProperty.Register("NumTrainingExamples", typeof(int), typeof(MonitorViewModel),
                new PropertyMetadata(1000));

        public static readonly DependencyProperty DayDreamsProperty = DependencyProperty.Register("DayDreams",
            typeof(ObservableCollection<BitmapSource>), typeof(MonitorViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty TrainingSetProperty = DependencyProperty.Register("TrainingSet",
            typeof(ObservableCollection<BitmapSource>), typeof(MonitorViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty ElapsedProperty = DependencyProperty.Register("Elapsed",
            typeof(TimeSpan), typeof(MonitorViewModel), new PropertyMetadata(default(TimeSpan)));

        public static readonly DependencyProperty ExitEvaluatorFactoryProperty =
            DependencyProperty.Register("ExitEvaluatorFactory", typeof(InteractiveExitEvaluatorFactory<double>),
                typeof(MonitorViewModel), new PropertyMetadata(default(InteractiveExitEvaluatorFactory<double>)));

        public static readonly DependencyProperty DisplayedEpochProperty = DependencyProperty.Register(
            "DisplayedEpoch", typeof(int), typeof(MonitorViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty WeightLearningRateFactoryProperty =
                   DependencyProperty.Register("WeightLearningRateFactory",
                       typeof(LayerSpecificLearningRateCalculatorFactory<double>), typeof(MonitorViewModel),
                       new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<double>)));

        public static readonly DependencyProperty DisplayFeatureCommandProperty =
            DependencyProperty.Register("DisplayFeatureCommand", typeof(ICommand), typeof(MonitorViewModel),
                new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty SelectedFeatureProperty =
            DependencyProperty.Register("SelectedFeature", typeof(BitmapSource), typeof(MonitorViewModel),
                new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty VisBiasLearningRateFactoryProperty =
            DependencyProperty.Register("VisBiasLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<double>), typeof(MonitorViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<double>)));

        public static readonly DependencyProperty HidBiasLearningRateFactoryProperty =
            DependencyProperty.Register("HidBiasLearningRateFactory",
                typeof(LayerSpecificLearningRateCalculatorFactory<double>), typeof(MonitorViewModel),
                new PropertyMetadata(default(LayerSpecificLearningRateCalculatorFactory<double>)));

        public static readonly DependencyProperty ErrorLabelBrushProperty =
            DependencyProperty.Register("ErrorLabelBrush", typeof(Brush), typeof(MonitorViewModel),
                new PropertyMetadata(default(Brush)));

        public static readonly DependencyProperty DeltaLabelBrushProperty =
            DependencyProperty.Register("DeltaLabelBrush", typeof(Brush), typeof(MonitorViewModel),
                new PropertyMetadata(default(Brush)));

        public static readonly DependencyProperty LayerConfigsProperty = DependencyProperty.Register("LayerConfigs",
            typeof(ObservableCollection<ConstructLayerBase>), typeof(MonitorViewModel),
            new PropertyMetadata(default(ObservableCollection<ConstructLayerBase>)));


        private CancellationTokenSource _cancelSource;


        public MonitorViewModel()
        {
            RunCommand = new CommandHandler(a => Run(), a => true);
        }

        public string DemoMode
        {
            get { return Dispatcher.InvokeIfRequired(() => (string)GetValue(DemoModeProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DemoModeProperty, value)).Wait(); }
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

        public ObservableCollection<ImagePair> Reconstructions
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () => (ObservableCollection<ImagePair>)GetValue(ReconstructionsProperty)).Result;
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
                var defineDlg = new DefineNetwork { Owner = Window.GetWindow(this) };
                var builder = defineDlg.DataContext as LayerBuilderViewModel;
                LayerConfigs = builder.LayerConstructionInfo;
                switch (DemoMode)
                {
                    case "Faces":
                        {
                            NumTrainingExamples = NumTrainingExamples > 0
                                ? NumTrainingExamples
                                : (NumTrainingExamples = 5000);
                            ConfigureDefaultFacesLayers(builder);

                            defineDlg.ShowDialog();
                            t = Task.Run(() => FacesDemo(builder, NumTrainingExamples, pathBase), _cancelSource.Token);
                            break;
                        }
                    case "Data":
                        {
                            ConfigureDefaultDataLayers(builder);

                            defineDlg.ShowDialog();

                            NumTrainingExamples = NumTrainingExamples > 0
                                ? NumTrainingExamples
                                : (NumTrainingExamples = 185945);
                            t = Task.Run(() => CsvDemo(builder, NumTrainingExamples, pathBase), _cancelSource.Token);
                            break;
                        }
                    case "Kaggle":
                        {
                            ConfigureDefaultKaggleLayers(builder);

                            defineDlg.ShowDialog();

                            //int numexamples = 40000;
                            NumTrainingExamples = NumTrainingExamples > 0
                                ? NumTrainingExamples
                                : (NumTrainingExamples = 40000);

                            t = Task.Run(() => KaggleDemo(builder, NumTrainingExamples, pathBase), _cancelSource.Token);
                            break;
                        }
                }

            }
            catch (TaskCanceledException)
            {

            }
            catch (OperationCanceledException)
            {

            }
        }

        private void ConfigureDefaultKaggleLayers(LayerBuilderViewModel builderViewModel)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 784,
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
                NumVisibleNeurons = 510,
                NumHiddenNeurons = 2000,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = 0.01
            });
        }

        private void ConfigureDefaultDataLayers(LayerBuilderViewModel builderViewModel)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructBinaryLayer
            {
                NumVisibleNeurons = 178,
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
                NumVisibleNeurons = 500,
                NumHiddenNeurons = 50,
                ConvertActivationsToStates = true,
                WeightInitializationStDev = 0.01
            });
        }

        private void ConfigureDefaultFacesLayers(LayerBuilderViewModel builderViewModel)
        {
            builderViewModel.LayerConstructionInfo.Add(new ConstructLinearHiddenLayer
            {
                NumVisibleNeurons = 75 * 75,
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
                NumVisibleNeurons = 4000,
                NumHiddenNeurons = 4000,
                WeightInitializationStDev = 0.01
            });
        }

        private async void CsvDemo(LayerBuilderViewModel layerBuilderViewModel, int numTrainingExamples, string pathBase)
        {
            GPGPU dev;
            GPGPURAND rand;
            InitCuda(out dev, out rand);
            dev.SetCurrentContext();
            IDataIO<TElement, string> d = new CsvData(ConfigurationManager.AppSettings["CsvDataTraining"],
                ConfigurationManager.AppSettings["CsvDataTest"], true, true);


            using (var net = new CudaAdvancedNetwork(layerBuilderViewModel.CreateLayers(dev, rand)))
            {
                net.SetDefaultMachineState(SuspendState.Active);
                string[] lbl;
                TElement[,] coded;

                double[,] tdata = d.ReadTestData(0, 50);
                List<double[,]> identityMatrices = IdentityMatrices(dev, net);

                IList<BitmapSource> originalTestImages = await GenerateImageSources(tdata);

                Dispatcher.InvokeIfRequired(
                    () =>
                        Reconstructions =
                            new ObservableCollection<ImagePair>(originalTestImages.Select(a => new ImagePair { Item1 = a })));

                dev.SetCurrentContext();

                net.EpochComplete += NetOnEpochComplete(pathBase, dev, tdata, identityMatrices);

                net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

                //batch the data in gpu memory
                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                {
                    ExitEvaluatorFactory =
                        await
                            Dispatcher.InvokeIfRequired(
                                () => new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000));

                    string[] lbla;
                    TElement[,] codeda;
                    double[,] trainingData = d.ReadTrainingData(0, numTrainingExamples, out lbla, out codeda);
                    Dispatcher.InvokeIfRequired(
                        async () =>
                            TrainingSet =
                                new ObservableCollection<BitmapSource>(await GenerateImageSources(trainingData, 1000)));

                    Dispatcher.InvokeIfRequired(() => NumTrainingExamples = trainingData.GetLength(0));

                    await Dispatcher.InvokeIfRequired(() =>
                    {
                        WeightLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(layerBuilderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05))); ;
                        HidBiasLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(layerBuilderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
                        VisBiasLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(layerBuilderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
                    });

                    //var trainingData = d.ReadTestData(0, numTrainingExamples);
                    dev.SetCurrentContext();
                    net.GreedyBatchedTrain(trainingData,
                        600,
                        ExitEvaluatorFactory,
                        WeightLearningRateFactory,
                        HidBiasLearningRateFactory,
                        VisBiasLearningRateFactory,
                        _cancelSource.Token
                        );
                }

                //double[,] testData = d.ReadTrainingData(0, 200, out lbl, out coded);

                //double[,] reconstructions = net.Reconstruct(testData);

                //DisplayResults(pathBase, d, reconstructions, testData, lbl);

                //IDataIO<TElement, string> d2 = new CsvData(ConfigurationManager.AppSettings["CsvDataTest"],
                //    ConfigurationManager.AppSettings["CsvDataTest"], true, true);

                //string[] labels;
                //TElement[,] lcoded;
                //double[,] allDataToCode = d2.ReadTrainingData(0, 185945, out labels, out lcoded);
                //double[,] encoded = net.Encode(allDataToCode);
                //string[] kkey = KeyEncoder.CreateBinaryStringKeys(encoded);

                //using (FileStream fs = File.OpenWrite(Path.Combine(pathBase, "Encoded.csv")))
                //using (var tw = new StreamWriter(fs))
                //{
                //    for (int i = 0; i < allDataToCode.GetLength(0); i++)
                //    {
                //        tw.WriteLine("{0},\"{1}\"", labels[i], kkey[i]);
                //    }
                //}
            }
        }

        private EventHandler<EpochEventArgs<double>> NetOnEpochComplete(string pathBase, GPGPU dev, double[,] tdata,
            List<double[,]> identityMatrices)
        {
            return async (a, b) =>
            {
                var nn = ((ICudaNetwork<TElement>)a);
                IAdvancedRbmCuda<double> m = nn.Machines[b.Layer];
                if (b.Epoch > 0 && b.Epoch % BackupFrequency == 0)
                {
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                            m.NumHiddenNeurons,
                            typeof(TElement).Name, b.Epoch)));
                }

                double[,] activations = GetActivations(dev, nn, tdata, b);

                if (b.Epoch % UpdateFrequency == 0)
                {
                    double[,] dreams = ((CudaAdvancedNetwork)nn).Daydream(1.0, 100, b.Layer);
                    double[,] recon = nn.Reconstruct(tdata, b.Layer);
                    double[,] feats = nn.Decode(identityMatrices[b.Layer], b.Layer);


                    Task.Run(() => UpdateUIProperties(pathBase, b, recon, feats, activations, dreams,
                        dd => GenerateImageSources(dd)));
                }
                else
                {
                    Task.Run(
                        () => UpdateUIProperties(pathBase, b, activations, dd => GenerateImageSourcesPosNeg(dd)));
                }
            };
        }

        private static EventHandler<EpochEventArgs<double>> NetOnLayerTrainComplete(string pathBase)
        {
            return (a, b) =>
            {
                IAdvancedRbmCuda<double> m = ((ICudaNetwork<TElement>)a).Machines[b.Layer];
                m.Save(Path.Combine(pathBase,
                    string.Format("Layer_{0}_{1}x{2}_{3}_Final.dat", b.Layer, m.NumVisibleNeurons,
                        m.NumHiddenNeurons,
                        typeof(TElement).Name)));
            };
        }


        private async void FacesDemo(LayerBuilderViewModel builderViewModel, int numTrainingExamples, string pathBase)
        {
            GPGPU dev;
            GPGPURAND rand;
            InitCuda(out dev, out rand);

            dev.SetCurrentContext();
            bool useLinear = builderViewModel.LayerConstructionInfo[0] is ConstructLinearHiddenLayer;

            IDataIO<TElement, string> dataProvider =
                new FacesData(ConfigurationManager.AppSettings["FacesDirectory"],
                    ConfigurationManager.AppSettings["FacesTestDirectory"],
                    FacesData.ConversionMode.RgbToGreyPosNegReal);


            Func<TElement[,], Task<IList<BitmapSource>>> imgGenerationMethod = useLinear
                ? (Func<TElement[,], Task<IList<BitmapSource>>>)(dd => GenerateImageSourcesPosNeg(dd))
                : (dd => GenerateImageSources(dd));

            using (var net = new CudaAdvancedNetwork(builderViewModel.CreateLayers(dev, rand)))
            {
                net.SetDefaultMachineState(SuspendState.Suspended);
                //keep data in main memory as much as possible at the expense of more memory movement between System and GPU

                double[,] tdata = dataProvider.ReadTestData(numTrainingExamples, 50);
                DirectoryInfo di = Directory.CreateDirectory(Path.Combine(pathBase, "Original"));

                dev.SetCurrentContext();
                List<double[,]> identityMatrices = IdentityMatrices(dev, net);

                Task.Run(() => Dispatcher.InvokeIfRequired(async () =>
                    Reconstructions =
                        new ObservableCollection<ImagePair>(
                            (await imgGenerationMethod(tdata)).Select(a => new ImagePair { Item1 = a }))));

                dev.SetCurrentContext();

                net.EpochComplete += NetOnEpochComplete(pathBase, dev, tdata, identityMatrices);
                net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

                IList<string[]> lbl;
                IList<TElement[,]> coded;

                IList<double[,]> training = dataProvider.ReadTrainingData(0, numTrainingExamples, 50, out lbl,
                    out coded);

                Dispatcher.InvokeIfRequired(() => NumTrainingExamples = training.Sum(a => a.GetLength(0)));

                //await (() => NumTrainingExamples = training.Sum(a => a.GetLength(0))).InvokeIfRequired(Dispatcher);

                int maxtrain = 1000;

                var bmps = new List<BitmapSource>(maxtrain);


                Task<List<BitmapSource>> tgen = Task.Run(async () =>
                {
                    foreach (var batch in training)
                    {
                        bmps.AddRange(await GenerateImageSourcesPosNeg(batch, maxtrain - bmps.Count));
                        if (bmps.Count >= maxtrain)
                            break;
                    }
                    return bmps;
                });

                Task.Run(
                    () =>
                        Dispatcher.InvokeIfRequired(
                            async () => TrainingSet = new ObservableCollection<BitmapSource>(await tgen)));


                dev.SetCurrentContext();

                await Dispatcher.InvokeIfRequired(() =>
                {
                    WeightLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(builderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05))); ;
                    HidBiasLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(builderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
                    VisBiasLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(builderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
                });

                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                {
                    ExitEvaluatorFactory =
                        await
                            Dispatcher.InvokeIfRequired(
                                () => new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000));

                    dev.SetCurrentContext();
                    net.GreedyBatchedTrainMem(training,
                        ExitEvaluatorFactory,
                        WeightLearningRateFactory,
                        HidBiasLearningRateFactory,
                        VisBiasLearningRateFactory,
                        _cancelSource.Token
                        );
                }
            }
        }


        private async void KaggleDemo(LayerBuilderViewModel builderViewModel, int numTrainingExamples, string pathBase)
        {
            GPGPU dev;
            GPGPURAND rand;
            InitCuda(out dev, out rand);
            dev.SetCurrentContext();
            IDataIO<TElement, int> dataProvider =
                new KaggleData(ConfigurationManager.AppSettings["KaggleTrainingData"],
                    ConfigurationManager.AppSettings["KaggleTestData"]);


            using (var net = new CudaAdvancedNetwork(builderViewModel.CreateLayers(dev, rand)))
            {
                //keep data in gpu memory as much as possible
                net.SetDefaultMachineState(SuspendState.Active);


                int[] lbl;
                TElement[,] coded;
                double[,] tdata = dataProvider.ReadTestData(0, 50);
                DirectoryInfo di = Directory.CreateDirectory(Path.Combine(pathBase, "Original"));
                List<double[,]> identityMatrices = IdentityMatrices(dev, net);

                Task.Run(() => Dispatcher.InvokeIfRequired(async () =>
                    Reconstructions =
                        new ObservableCollection<ImagePair>(
                            (await GenerateImageSources(tdata)).Select(a => new ImagePair { Item1 = a }))));

                dev.SetCurrentContext();

                net.EpochComplete += NetEpochCompleteCodingEventHandler(pathBase, tdata, identityMatrices, dev);
                net.LayerTrainComplete += NetOnLayerTrainComplete(pathBase);

                double[,] trainingData = dataProvider.ReadTrainingData(0, numTrainingExamples, out lbl, out coded);
                Task.Run(() =>
                    Dispatcher.InvokeIfRequired(
                        async () =>
                            TrainingSet =
                                new ObservableCollection<BitmapSource>(await GenerateImageSources(trainingData, 1000))));

                Dispatcher.InvokeIfRequired(() => NumTrainingExamples = trainingData.GetLength(0));

                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                {
                    ExitEvaluatorFactory =
                        await
                            Dispatcher.InvokeIfRequired(
                                () => new InteractiveExitEvaluatorFactory<double>(greedyTracker, 0.5, 5000));


                    await Dispatcher.InvokeIfRequired(() =>
                    {
                        WeightLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(builderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05))); ;
                        HidBiasLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(builderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
                        VisBiasLearningRateFactory = new LayerSpecificLearningRateCalculatorFactory<double>(builderViewModel.LayerConstructionInfo.Select(a => new InteractiveLearningRateCalculatorFactory<double>(3E-05)));
                    });

                    dev.SetCurrentContext();
                    net.GreedyBatchedSupervisedTrain(
                        trainingData,
                        coded, 100,
                        ExitEvaluatorFactory,
                        WeightLearningRateFactory,
                        HidBiasLearningRateFactory,
                        VisBiasLearningRateFactory,
                        _cancelSource.Token
                        );
                }
            }
        }

        private EventHandler<EpochEventArgs<double>> NetEpochCompleteCodingEventHandler(string pathBase, double[,] tdata,
            List<double[,]> identityMatrices, GPGPU dev)
        {
            return async (a, b) =>
            {
                var nn = ((ICudaNetwork<TElement>)a);
                IAdvancedRbmCuda<double> m = nn.Machines[b.Layer];
                if (b.Epoch > 0 && b.Epoch % BackupFrequency == 0)
                {
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                            m.NumHiddenNeurons,
                            typeof(TElement).Name, b.Epoch)));
                }

                if (b.Epoch % UpdateFrequency == 0)
                {
                    double[,] dreams;
                    double[,] recon;
                    double[,] feats;
                    double[,] activations;
                    if (b.Layer == nn.Machines.Count - 1)
                    {
                        double[,] lbls;
                        dreams = ((CudaAdvancedNetwork)nn).DaydreamWithLabels(1.0, 100, out lbls);
                        double[,] llbl;
                        recon = nn.ReconstructWithLabels(tdata, out llbl);
                        double[,] llbls;
                        feats = nn.DecodeWithLabels(identityMatrices[b.Layer], out llbls);
                        activations = GetActivationsWithLabelExpansion(dev, nn, tdata);
                    }
                    else
                    {
                        dreams = ((CudaAdvancedNetwork)nn).Daydream(1.0, 100, b.Layer);
                        recon = nn.Reconstruct(tdata, b.Layer);
                        feats = nn.Decode(identityMatrices[b.Layer], b.Layer);
                        activations = GetActivations(dev, nn, tdata, b);
                    }


                    Task.Run(() => UpdateUIProperties(pathBase, b, recon, feats, activations, dreams,
                        dd => GenerateImageSources(dd)));
                }
                else
                {
                    double[,] activations;
                    if (b.Layer == nn.Machines.Count - 1)
                    {
                        activations = GetActivationsWithLabelExpansion(dev, nn, tdata);
                    }
                    else
                    {
                        activations = GetActivations(dev, nn, tdata, b);
                    }
                    Task.Run(
                        () => UpdateUIProperties(pathBase, b, activations, dd => GenerateImageSourcesPosNeg(dd)));
                }
            };
        }


        private static double[,] GetActivations(GPGPU dev, ICudaNetwork<double> nn, double[,] tdata,
            EpochEventArgs<double> b)
        {
            TElement[,] activations;
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
            TElement[,] activations;
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
            var identityMatrices = new List<TElement[,]>();

            dev.SetCurrentContext();
            foreach (var advancedRbmCuda in net.Machines)
            {
                using (
                    Matrix2D<double> m = dev.AllocateNoSet<TElement>(advancedRbmCuda.NumHiddenNeurons,
                        advancedRbmCuda.NumHiddenNeurons))
                {
                    m.Identity();
                    identityMatrices.Add(m.CopyLocal());
                }
            }
            return identityMatrices;
        }

        private async Task UpdateUIProperties(string pathBase, EpochEventArgs<TElement> epochEventArgs,
            TElement[,] activations, Func<TElement[,], Task<IList<BitmapSource>>> bitmapConverter)
        {
            Epoch = epochEventArgs.Epoch;
            Error = epochEventArgs.Error;
            Layer = epochEventArgs.Layer;
            Delta = epochEventArgs.Delta;
            Elapsed = epochEventArgs.Elapsed;
            LearningRate = epochEventArgs.LearningRate;
            IList<BitmapSource> actiim = await bitmapConverter(activations);

            await Dispatcher.InvokeIfRequired(() => { ActivationFrequency = actiim[0]; });
        }

        private async void UpdateUIProperties(string pathBase, EpochEventArgs<double> b, double[,] recon,
            double[,] feats,
            double[,] activations, double[,] dreams, Func<TElement[,], Task<IList<BitmapSource>>> imgGenerator)
        {
            DisplayedEpoch = b.Epoch;
            Task t = UpdateUIProperties(pathBase, b, activations, imgGenerator);

            Task t1 = Task.Run(async () => UpdateImageResult(Reconstructions, await imgGenerator(recon)));
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

        private async void UpdateImageResult(ObservableCollection<ImagePair> set, IList<BitmapSource> reconIm)
        {
            await Dispatcher.InvokeIfRequired(() =>
            {
                for (int i = 0; i < reconIm.Count; i++)
                {
                    set[i].Item2 = reconIm[i];
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
                new ObservableCollection<ImagePair>(finalTest.Zip(finalRecon,
                    (a, b) => new ImagePair { Item1 = a, Item2 = b }));
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


        private async Task<IList<BitmapSource>> GenerateImageSourcesInt(TElement[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte)(b), maxResults);
        }

        private async Task<IList<BitmapSource>> GenerateImageSourcesPosNeg(
            TElement[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte)((b + 0.5) * 255.0), maxResults);
        }


        private static void InitCuda(out GPGPU dev, out GPGPURAND rand)
        {
            CudafyHost.ClearAllDeviceMemories();
            CudafyHost.ClearDevices();


            dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            GPGPUProperties props = dev.GetDeviceProperties();
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
                    DeltaLabelBrush = (double)e.OldValue > (double)e.NewValue
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