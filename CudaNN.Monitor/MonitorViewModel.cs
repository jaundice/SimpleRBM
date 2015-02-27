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
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using SimpleRBM.Common;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Common.LearningRate;
using SimpleRBM.Cuda;
using SimpleRBM.Demo;
using SimpleRBM.Demo.IO;
using SimpleRBM.Demo.Util;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;

#endif

namespace CudaNN.Monitor
{
    public static class DispatcherEx
    {
        public static async void InvokeIfRequired(this Dispatcher self, Action action)
        {
            if (self.CheckAccess())
                action();
            else
                await self.InvokeAsync(action);
        }

        public static async Task<T> InvokeIfRequired<T>(this Dispatcher self, Func<T> action)
        {
            if (self.CheckAccess())
                return action();
            return await self.InvokeAsync(action);
        }
    }

    public class MonitorViewModel : DependencyObject
    {
        public static readonly DependencyProperty DemoModeProperty = DependencyProperty.Register("DemoMode",
            typeof (string), typeof (MonitorViewModel), new PropertyMetadata(default(string)));

        public static readonly DependencyProperty LayerProperty = DependencyProperty.Register("Layer", typeof (int),
            typeof (MonitorViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty EpochProperty = DependencyProperty.Register("Epoch", typeof (int),
            typeof (MonitorViewModel), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty LearningRateProperty = DependencyProperty.Register("LearningRate",
            typeof (TElement), typeof (MonitorViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty UpdateFrequencyProperty =
            DependencyProperty.Register("UpdateFrequency", typeof (int), typeof (MonitorViewModel),
                new PropertyMetadata(default(int)));

        public static readonly DependencyProperty ActivationFrequencyProperty =
            DependencyProperty.Register("ActivationFrequency", typeof (BitmapSource), typeof (MonitorViewModel),
                new PropertyMetadata(default(BitmapSource)));

        public static readonly DependencyProperty ReconstructionsProperty =
            DependencyProperty.Register("Reconstructions", typeof (ObservableCollection<ImagePair>),
                typeof (MonitorViewModel), new PropertyMetadata(default(ObservableCollection<ImagePair>)));

        public static readonly DependencyProperty FeaturesProperty = DependencyProperty.Register("Features",
            typeof (ObservableCollection<BitmapSource>), typeof (MonitorViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty RunAppMethodProperty = DependencyProperty.Register("RunAppMethod",
            typeof (ICommand), typeof (MonitorViewModel), new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty RunBindingProperty = DependencyProperty.Register("RunBinding",
            typeof (CommandBinding), typeof (MonitorViewModel), new PropertyMetadata(default(CommandBinding)));

        public static readonly DependencyProperty RunCommandProperty = DependencyProperty.Register("RunCommand",
            typeof (ICommand), typeof (MonitorViewModel), new PropertyMetadata(default(ICommand)));

        public static readonly DependencyProperty BackupFrequencyProperty =
            DependencyProperty.Register("BackupFrequency", typeof (int), typeof (MonitorViewModel),
                new PropertyMetadata(default(int)));

        public static readonly DependencyProperty ErrorProperty = DependencyProperty.Register("Error", typeof (TElement),
            typeof (MonitorViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty DeltaProperty = DependencyProperty.Register("Delta", typeof (TElement),
            typeof (MonitorViewModel), new PropertyMetadata(default(TElement)));

        public static readonly DependencyProperty NumTrainingExamplesProperty =
            DependencyProperty.Register("NumTrainingExamples", typeof (int), typeof (MonitorViewModel),
                new PropertyMetadata(default(int)));

        public static readonly DependencyProperty DayDreamsProperty = DependencyProperty.Register("DayDreams",
            typeof (ObservableCollection<BitmapSource>), typeof (MonitorViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        public static readonly DependencyProperty TrainingSetProperty = DependencyProperty.Register("TrainingSet",
            typeof (ObservableCollection<BitmapSource>), typeof (MonitorViewModel),
            new PropertyMetadata(default(ObservableCollection<BitmapSource>)));

        private CancellationTokenSource _cancelSource;

        public MonitorViewModel()
        {
            RunCommand = new CommandHandler(a => Run(), a => true);
        }

        public string DemoMode
        {
            get { return Dispatcher.InvokeIfRequired(() => (string) GetValue(DemoModeProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DemoModeProperty, value)); }
        }

        public int Layer
        {
            get { return Dispatcher.InvokeIfRequired(() => (int) GetValue(LayerProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LayerProperty, value)); }
        }

        public int Epoch
        {
            get { return Dispatcher.InvokeIfRequired(() => (int) GetValue(EpochProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(EpochProperty, value)); }
        }

        public TElement LearningRate
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement) GetValue(LearningRateProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LearningRateProperty, value)); }
        }

        public int UpdateFrequency
        {
            get { return Dispatcher.InvokeIfRequired(() => (int) GetValue(UpdateFrequencyProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(UpdateFrequencyProperty, value)); }
        }

        public BitmapSource ActivationFrequency
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (BitmapSource) GetValue(ActivationFrequencyProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ActivationFrequencyProperty, value)); }
        }

        public ObservableCollection<ImagePair> Reconstructions
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(
                        () => (ObservableCollection<ImagePair>) GetValue(ReconstructionsProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ReconstructionsProperty, value)); }
        }

        public ObservableCollection<BitmapSource> Features
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (ObservableCollection<BitmapSource>) GetValue(FeaturesProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(FeaturesProperty, value)); }
        }

        public ICommand RunCommand
        {
            get { return Dispatcher.InvokeIfRequired(() => (ICommand) GetValue(RunCommandProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(RunCommandProperty, value)); }
        }

        public int BackupFrequency
        {
            get { return Dispatcher.InvokeIfRequired(() => (int) GetValue(BackupFrequencyProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(BackupFrequencyProperty, value)); }
        }

        public TElement Error
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement) GetValue(ErrorProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(ErrorProperty, value)); }
        }

        public TElement Delta
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement) GetValue(DeltaProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DeltaProperty, value)); }
        }

        public int NumTrainingExamples
        {
            get { return Dispatcher.InvokeIfRequired(() => (int) GetValue(NumTrainingExamplesProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(NumTrainingExamplesProperty, value)); }
        }

        public ObservableCollection<BitmapSource> DayDreams
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (ObservableCollection<BitmapSource>) GetValue(DayDreamsProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(DayDreamsProperty, value)); }
        }

        public ObservableCollection<BitmapSource> TrainingSet
        {
            get
            {
                return
                    Dispatcher.InvokeIfRequired(() => (ObservableCollection<BitmapSource>) GetValue(TrainingSetProperty))
                        .Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(TrainingSetProperty, value)); }
        }

        private async void Run()
        {
            string pathBase = Path.Combine(Environment.CurrentDirectory,
                string.Format("{0}_{1}", DateTime.Now.ToString("u").Replace(':', '-'), DemoMode));

            Directory.CreateDirectory(pathBase);
            UpdateFrequency = 20;
            BackupFrequency = 500;

            Task t = null;
            try
            {
                if (_cancelSource != null)
                    _cancelSource.Cancel();
            }
            catch (TaskCanceledException)
            {
            }

            _cancelSource = new CancellationTokenSource();

            switch (DemoMode)
            {
                case "Faces":
                {
                    int numexamples = 5000;
                    t = Task.Run(() => FacesDemo(numexamples, pathBase), _cancelSource.Token);
                    break;
                }
                case "Data":
                {
                    int numexamples = 185945;
                    t = Task.Run(() => CsvDemo(numexamples, pathBase), _cancelSource.Token);
                    break;
                }
                case "Kaggle":
                {
                    //int numexamples = 40000;
                    int numexamples = 4000;

                    t = Task.Run(() => KaggleDemo(numexamples, pathBase), _cancelSource.Token);
                    break;
                }
            }
        }

        private async void CsvDemo(int numTrainingExamples, string pathBase)
        {
            GPGPU dev;
            GPGPURAND rand;
            InitCuda(out dev, out rand);
            dev.SetCurrentContext();
            IDataIO<TElement, string> d = new CsvData(ConfigurationManager.AppSettings["CsvDataTraining"],
                ConfigurationManager.AppSettings["CsvDataTest"], true, true);


            using (var net = new CudaAdvancedNetwork(new CudaAdvancedRbmBase[]
            {
                new CudaAdvancedRbmBinary(dev, rand, 0, 178, 1000, false),
                new CudaAdvancedRbmBinary(dev, rand, 1, 1000, 1000, true),
                new CudaAdvancedRbmBinary(dev, rand, 2, 1000, 50, true)
            }))
            {
                net.SetDefaultMachineState(SuspendState.Active);
                string[] lbl;
                TElement[,] coded;

                double[,] tdata = d.ReadTestData(0, 50);
                List<double[,]> identityMatrices = IdentityMatrices(dev, net);

                IList<BitmapSource> originalTestImages =
                    await Task.Run(async () => await GenerateImageSources(tdata));
                Reconstructions = await Dispatcher.InvokeIfRequired(() =>
                    new ObservableCollection<ImagePair>(originalTestImages.Select(a => new ImagePair {Item1 = a})));

                dev.SetCurrentContext();

                net.EpochComplete += async (a, b) =>
                {
                    var nn = ((ICudaNetwork<TElement>) a);
                    IAdvancedRbmCuda<double> m = nn.Machines[b.Layer];
                    if (b.Epoch > 0 && b.Epoch%BackupFrequency == 0)
                    {
                        m.Save(Path.Combine(pathBase,
                            string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                                m.NumHiddenNeurons,
                                typeof (TElement).Name, b.Epoch)));
                    }

                    if (b.Epoch%UpdateFrequency == 0)
                    {
                        double[,] dreams = ((CudaAdvancedNetwork) nn).Daydream(1.0, 100, b.Layer);
                        double[,] recon = nn.Reconstruct(tdata, b.Layer);
                        double[,] feats = nn.Decode(identityMatrices[b.Layer], b.Layer);

                        TElement[,] activations;
                        using (Matrix2D<double> enc = dev.Upload(nn.Encode(tdata, b.Layer)))
                        using (Matrix2D<double> act = enc.SumRows())
                        using (Matrix2D<double> trans = act.Transpose())
                        using (Matrix2D<double> max = trans.MaxRowValues())
                        using (Matrix2D<double> sm = trans.DivideElements(max))
                        {
                            //activations = act.CopyLocal();
                            activations = sm.CopyLocal();
                        }


                        UpdateUIProperties(pathBase, b, recon, feats, activations, dreams,
                            dd => GenerateImageSources(dd));
                    }
                };

                net.LayerTrainComplete += (a, b) =>
                {
                    IAdvancedRbmCuda<double> m = ((ICudaNetwork<TElement>) a).Machines[b.Layer];
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}_Final.dat", b.Layer, m.NumVisibleNeurons,
                            m.NumHiddenNeurons,
                            typeof (TElement).Name)));
                };

                //batch the data in gpu memory
                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                {
                    string[] lbla;
                    TElement[,] codeda;
                    double[,] trainingData = d.ReadTrainingData(0, numTrainingExamples, out lbla, out codeda);
                    Dispatcher.InvokeIfRequired(
                        async () =>
                            TrainingSet =
                                new ObservableCollection<BitmapSource>(await GenerateImageSources(trainingData, 1000)));

                    await Dispatcher.InvokeIfRequired(() => NumTrainingExamples = trainingData.GetLength(0));

                    //var trainingData = d.ReadTestData(0, numTrainingExamples);
                    dev.SetCurrentContext();
                    net.GreedyBatchedTrain(trainingData,
                        600,
                        new EpochCountExitConditionFactory<TElement>(greedyTracker, 5000),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.0001, 20),
                            new ConstantLearningRateFactory<TElement>(0.00005),
                            new ConstantLearningRateFactory<TElement>(0.000001)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.0001, 20),
                            new ConstantLearningRateFactory<TElement>(0.00005),
                            new ConstantLearningRateFactory<TElement>(0.000001)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.0001, 20),
                            new ConstantLearningRateFactory<TElement>(0.00005),
                            new ConstantLearningRateFactory<TElement>(0.000001))
                        );
                }

                double[,] testData = d.ReadTrainingData(0, 200, out lbl, out coded);

                double[,] reconstructions = net.Reconstruct(testData);

                DisplayResults(pathBase, d, reconstructions, testData, lbl);

                IDataIO<TElement, string> d2 = new CsvData(ConfigurationManager.AppSettings["CsvDataTest"],
                    ConfigurationManager.AppSettings["CsvDataTest"], true, true);

                string[] labels;
                TElement[,] lcoded;
                double[,] allDataToCode = d2.ReadTrainingData(0, 185945, out labels, out lcoded);
                double[,] encoded = net.Encode(allDataToCode);
                string[] kkey = KeyEncoder.CreateBinaryStringKeys(encoded);

                using (FileStream fs = File.OpenWrite(Path.Combine(pathBase, "Encoded.csv")))
                using (var tw = new StreamWriter(fs))
                {
                    for (int i = 0; i < allDataToCode.GetLength(0); i++)
                    {
                        tw.WriteLine("{0},\"{1}\"", labels[i], kkey[i]);
                    }
                }
            }
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

        private async Task UpdateUIProperties(string pathBase, EpochEventArgs<double> b, double[,] recon,
            double[,] feats,
            double[,] activations, double[,] dreams, Func<TElement[,], Task<IList<BitmapSource>>> imgSaver)
        {
            Epoch = b.Epoch;
            Error = b.Error;
            Layer = b.Layer;
            IList<BitmapSource> reconIm = await imgSaver(
                recon);
            IList<BitmapSource> featIm = await imgSaver(
                feats);
            IList<BitmapSource> actiim = await imgSaver(
                activations);


            UpdateImageResult(Reconstructions, reconIm);

            IList<BitmapSource> dreamIm =
                await imgSaver(dreams);

            await Dispatcher.InvokeIfRequired(async () =>
            {
                var feat = new ObservableCollection<BitmapSource>(featIm);
                Features = feat;
                DayDreams = new ObservableCollection<BitmapSource>(dreamIm);
                ActivationFrequency = actiim[0];
            });
        }

        private void UpdateImageResult(ObservableCollection<ImagePair> set, IList<BitmapSource> reconIm)
        {
            Dispatcher.InvokeIfRequired(() =>
            {
                for (int i = 0; i < reconIm.Count; i++)
                {
                    set[i].Item2 = reconIm[i];
                }
            });
        }


        private async void FacesDemo(int numTrainingExamples, string pathBase)
        {
            GPGPU dev;
            GPGPURAND rand;
            InitCuda(out dev, out rand);

            dev.SetCurrentContext();
            bool useLinear = true;

            IDataIO<TElement, string> dataProvider =
                new FacesData(ConfigurationManager.AppSettings["FacesDirectory"],
                    ConfigurationManager.AppSettings["FacesTestDirectory"],
                    FacesData.ConversionMode.RgbToGreyPosNegReal);


            Func<TElement[,], Task<IList<BitmapSource>>> imageSaveMethod = useLinear
                ? (Func<TElement[,], Task<IList<BitmapSource>>>) (dd => GenerateImageSourcesPosNeg(dd))
                : (dd => GenerateImageSources(dd));

            using (var net = new CudaAdvancedNetwork(useLinear
                ? new CudaAdvancedRbmBase[]
                {
                    new CudaAdvancedRbmLinearHidden(dev, rand, 0, 250*250, 500, 0.02),
                    new CudaAdvancedRbmLinearHidden(dev, rand, 0, 500, 4000, 0.02),
                    new CudaAdvancedRbmLinearHidden(dev, rand, 0, 4000, 4000, 0.02)
                }
                : new CudaAdvancedRbmBase[]
                {
                    new CudaAdvancedRbmBinary(dev, rand, 0, 250*250, 200, false),
                    new CudaAdvancedRbmBinary(dev, rand, 1, 200, 4000, true),
                    new CudaAdvancedRbmBinary(dev, rand, 2, 4000, 4000, true)
                }))
            {
                net.SetDefaultMachineState(SuspendState.Suspended);
                //keep data in main memory as much as possible at the expense of more memory movement between System and GPU

                double[,] tdata = dataProvider.ReadTestData(0, 50);
                DirectoryInfo di = Directory.CreateDirectory(Path.Combine(pathBase, "Original"));

                dev.SetCurrentContext();
                List<double[,]> identityMatrices = IdentityMatrices(dev, net);


                IList<BitmapSource> originalTestImages =
                    await Task.Run(async () => await GenerateImageSourcesPosNeg(tdata));
                Reconstructions = await Dispatcher.InvokeIfRequired(() =>
                    new ObservableCollection<ImagePair>(originalTestImages.Select(a => new ImagePair {Item1 = a})));

                dev.SetCurrentContext();

                net.EpochComplete += async (a, b) =>
                {
                    var nn = ((ICudaNetwork<TElement>) a);
                    IAdvancedRbmCuda<double> m = nn.Machines[b.Layer];
                    if (b.Epoch > 0 && b.Epoch%BackupFrequency == 0)
                    {
                        m.Save(Path.Combine(pathBase,
                            string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                                m.NumHiddenNeurons,
                                typeof (TElement).Name, b.Epoch)));
                    }

                    if (b.Epoch%UpdateFrequency == 0)
                    {
                        double[,] dreams = ((CudaAdvancedNetwork) nn).Daydream(1.0, 100, b.Layer);
                        double[,] recon = nn.Reconstruct(tdata, b.Layer);
                        double[,] feats = nn.Decode(identityMatrices[b.Layer], b.Layer);

                        TElement[,] activations;
                        using (Matrix2D<double> enc = dev.Upload(nn.Encode(tdata, b.Layer)))
                        using (Matrix2D<double> act = enc.SumRows())
                        using (Matrix2D<double> trans = act.Transpose())
                        using (Matrix2D<double> max = trans.MaxRowValues())
                        using (Matrix2D<double> sm = trans.DivideElements(max))
                        {
                            //activations = act.CopyLocal();
                            activations = sm.CopyLocal();
                        }


                        await
                            Task.Run(
                                async () =>
                                    await
                                        UpdateUIProperties(pathBase, b, recon, feats, activations, dreams,
                                            dd => GenerateImageSourcesPosNeg(dd)));
                    }
                };

                net.LayerTrainComplete += (a, b) =>
                {
                    IAdvancedRbmCuda<double> m = ((ICudaNetwork<TElement>) a).Machines[b.Layer];
                    m.Save(Path.Combine(pathBase,
                        string.Format("Layer_{0}_{1}x{2}_{3}.dat", b.Layer, m.NumVisibleNeurons, m.NumHiddenNeurons,
                            typeof (TElement).Name)));
                };

                IList<string[]> lbl;
                IList<TElement[,]> coded;
                IList<double[,]> training = dataProvider.ReadTrainingData(0, numTrainingExamples, 10, out lbl,
                    out coded);
                await
                    Dispatcher.InvokeIfRequired(
                        () =>
                            TrainingSet =
                                new ObservableCollection<BitmapSource>(
                                    training.Select(a => GenerateImageSourcesPosNeg(a).Result).SelectMany(b => b)));

                await Dispatcher.InvokeIfRequired(() => NumTrainingExamples = training.Sum(a => a.GetLength(0)));
                dev.SetCurrentContext();
                //batch the data into main memory
                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                {
                    net.GreedyBatchedTrainMem(training,
                        new EpochCountExitConditionFactory<TElement>(greedyTracker, 5000),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.000003, 20),
                            new ConstantLearningRateFactory<TElement>(0.000003),
                            new ConstantLearningRateFactory<TElement>(0.000003)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.000003, 20),
                            new ConstantLearningRateFactory<TElement>(0.000003),
                            new ConstantLearningRateFactory<TElement>(0.000003)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.000003, 20),
                            new ConstantLearningRateFactory<TElement>(0.000003),
                            new ConstantLearningRateFactory<TElement>(0.000003))
                        );
                }
            }
        }


        private async void KaggleDemo(int numTrainingExamples, string pathBase)
        {
            GPGPU dev;
            GPGPURAND rand;
            InitCuda(out dev, out rand);
            dev.SetCurrentContext();
            IDataIO<TElement, int> dataProvider =
                new KaggleData(ConfigurationManager.AppSettings["KaggleTrainingData"],
                    ConfigurationManager.AppSettings["KaggleTestData"]);

            using (var net = new CudaAdvancedNetwork(new CudaAdvancedRbmBase[]
            {
                new CudaAdvancedRbmBinary(dev, rand, 0, 784, 500, false),
                new CudaAdvancedRbmBinary(dev, rand, 1, 500, 500, true),
                new CudaAdvancedRbmBinary(dev, rand, 2, 510, 2000, true)
                //visible buffer expanded by 10 for labeling
            }))
            {
                //keep data in gpu memory as much as possible
                net.SetDefaultMachineState(SuspendState.Active);


                int[] lbl;
                TElement[,] coded;
                double[,] tdata = dataProvider.ReadTestData(0, 50);
                DirectoryInfo di = Directory.CreateDirectory(Path.Combine(pathBase, "Original"));
                List<double[,]> identityMatrices = IdentityMatrices(dev, net);


                IList<BitmapSource> originalTestImages =
                    await Task.Run(async () => await GenerateImageSources(tdata));
                Reconstructions = await Dispatcher.InvokeIfRequired(() =>
                    new ObservableCollection<ImagePair>(originalTestImages.Select(a => new ImagePair {Item1 = a})));

                dev.SetCurrentContext();

                net.EpochComplete += async (a, b) =>
                {
                    var nn = ((ICudaNetwork<TElement>) a);
                    IAdvancedRbmCuda<double> m = nn.Machines[b.Layer];
                    if (b.Epoch > 0 && b.Epoch%BackupFrequency == 0)
                    {
                        m.Save(Path.Combine(pathBase,
                            string.Format("Layer_{0}_{1}x{2}_{3}_Temp_{4}.dat", b.Layer, m.NumVisibleNeurons,
                                m.NumHiddenNeurons,
                                typeof (TElement).Name, b.Epoch)));
                    }

                    if (b.Epoch%UpdateFrequency == 0)
                    {
                        double[,] dreams = ((CudaAdvancedNetwork) nn).Daydream(1.0, 100, b.Layer);
                        double[,] recon = nn.Reconstruct(tdata, b.Layer);
                        double[,] feats = nn.Decode(identityMatrices[b.Layer], b.Layer);

                        TElement[,] activations;
                        using (Matrix2D<double> enc = dev.Upload(nn.Encode(tdata, b.Layer)))
                        using (Matrix2D<double> act = enc.SumRows())
                        using (Matrix2D<double> trans = act.Transpose())
                        using (Matrix2D<double> max = trans.MaxRowValues())
                        using (Matrix2D<double> sm = trans.DivideElements(max))
                        {
                            //activations = act.CopyLocal();
                            activations = sm.CopyLocal();
                        }


                        await
                            Task.Run(
                                async () =>
                                    await
                                        UpdateUIProperties(pathBase, b, recon, feats, activations, dreams,
                                            dd => GenerateImageSources(dd)));
                    }
                };

                double[,] trainingData = dataProvider.ReadTrainingData(0, numTrainingExamples, out lbl, out coded);
                Dispatcher.InvokeIfRequired(
                    async () =>
                        TrainingSet =
                            new ObservableCollection<BitmapSource>(await GenerateImageSources(trainingData, 1000)));

                await Dispatcher.InvokeIfRequired(() => NumTrainingExamples = trainingData.GetLength(0));

                dev.SetCurrentContext();
                using (
                    var greedyTracker =
                        new EpochErrorFileTracker<TElement>(Path.Combine(pathBase, "GreedyTrainError.log")))
                    net.GreedyBatchedSupervisedTrain(
                        trainingData,
                        coded, 100,
                        new EpochCountExitConditionFactory<TElement>(greedyTracker, 5000),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.003, 20),
                            new ConstantLearningRateFactory<TElement>(0.0005),
                            new ConstantLearningRateFactory<TElement>(0.0001)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.003, 20),
                            new ConstantLearningRateFactory<TElement>(0.0005),
                            new ConstantLearningRateFactory<TElement>(0.0001)),
                        new LayerSpecificLearningRateCalculatorFactory<TElement>(
                            new ConstantLearningRateFactory<TElement>(0.003, 20),
                            new ConstantLearningRateFactory<TElement>(0.0005),
                            new ConstantLearningRateFactory<TElement>(0.0001))
                        );
                int[] testSrcLabels;
                TElement[,] testSourceCoded;
                double[,] testData = dataProvider.ReadTrainingData(numTrainingExamples, 500, out testSrcLabels,
                    out testSourceCoded);

                TElement[,] computedLabels;
                double[,] reconstructions = net.ReconstructWithLabels(testData, out computedLabels, softmaxLabels: true);
                Console.WriteLine("Reconstructions");
                DisplayResults(pathBase, dataProvider, reconstructions, testData, testSrcLabels, testSourceCoded,
                    computedLabels);
                Console.WriteLine("Daydream by class");
            }
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
                    (a, b) => new ImagePair {Item1 = a, Item2 = b}));
        }

        private async Task<IList<BitmapSource>> GenerateImageSources(TElement[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte) (b*255f), maxResults);
        }

        private async Task<IList<BitmapSource>> GenerateImageSources(TElement[,] data,
            Func<TElement, byte> converter, int maxResults)
        {
            return await Dispatcher.InvokeIfRequired(() =>
            {
                var images = new BitmapSource[data.GetLength(0)];
                Parallel.For(0, Math.Min(data.GetLength(0), maxResults),
                    async a =>
                    {
                        int stride;
                        Bitmap bmp = ImageUtils.GenerateBitmap(data, a, converter, out stride);
                        IntPtr h = bmp.GetHbitmap();
                        try
                        {
                            await
                                Dispatcher.InvokeIfRequired(
                                    () =>
                                        images[a] =
                                            Imaging.CreateBitmapSourceFromHBitmap(h, IntPtr.Zero, Int32Rect.Empty,
                                                BitmapSizeOptions.FromEmptyOptions()));
                        }
                        finally
                        {
                            DeleteObject(h);
                        }
                    });
                return images;
            });
        }

        [DllImport("gdi32", EntryPoint = "DeleteObject")]
        private static extern int DeleteObject(IntPtr o);


        private async Task<IList<BitmapSource>> GenerateImageSourcesInt(TElement[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte) (b), maxResults);
        }

        private async Task<IList<BitmapSource>> GenerateImageSourcesPosNeg(
            TElement[,] data, int maxResults = int.MaxValue)
        {
            return await GenerateImageSources(data, b => (byte) ((b + 0.5)*255.0), maxResults);
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
                    typeof (ActivationFunctionsCuda),
                    typeof (Matrix2DCuda)
                    );
                Console.WriteLine("Saving kernels to {0}", kernelPath);
                mod.Serialize(kernelPath);
            }

            ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount,
                props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            rand = GPGPURAND.Create(dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

            rand.SetPseudoRandomGeneratorSeed((ulong) DateTime.Now.Ticks);
            rand.GenerateSeeds();

            Console.WriteLine("Loading Module");
            dev.LoadModule(mod);
        }
    }
}