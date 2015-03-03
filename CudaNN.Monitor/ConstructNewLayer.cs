using System.Windows;

namespace CudaNN.Monitor
{
    public class ConstructNewLayer : ConstructLayerBase
    {
        public static readonly DependencyProperty LayerIndexProperty =
            DependencyProperty.Register("LayerIndex", typeof(int),
                typeof(ConstructNewLayer), new PropertyMetadata(default(int)));

        public static readonly DependencyProperty NumVisibleNeuronsProperty =
            DependencyProperty.Register("NumVisibleNeurons", typeof(int),
                typeof(ConstructNewLayer), new PropertyMetadata(500));

        public static readonly DependencyProperty NumHiddenNeuronsProperty =
            DependencyProperty.Register("NumHiddenNeurons", typeof(int),
                typeof(ConstructNewLayer), new PropertyMetadata(500));

        public static readonly DependencyProperty WeightCostProperty =
            DependencyProperty.Register("WeightCost", typeof(double),
                typeof(ConstructNewLayer), new PropertyMetadata(2E-04));

        public static readonly DependencyProperty InitialMomentumProperty =
            DependencyProperty.Register("InitialMomentum", typeof(double),
                typeof(ConstructNewLayer), new PropertyMetadata(0.5));


        public static readonly DependencyProperty FinalMomentumProperty =
            DependencyProperty.Register("FinalMomentum", typeof(double),
                typeof(ConstructNewLayer), new PropertyMetadata(0.9));

        public static readonly DependencyProperty WeightInitializationStDevProperty =
            DependencyProperty.Register("WeightInitializationStDev", typeof(double),
                typeof(ConstructNewLayer), new PropertyMetadata(0.01));

        public int LayerIndex
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(LayerIndexProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(LayerIndexProperty, value)).Wait(); }
        }

        public int NumVisibleNeurons
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(NumVisibleNeuronsProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(NumVisibleNeuronsProperty, value)).Wait(); }
        }

        public int NumHiddenNeurons
        {
            get { return Dispatcher.InvokeIfRequired(() => (int)GetValue(NumHiddenNeuronsProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(NumHiddenNeuronsProperty, value)).Wait(); }
        }

        public double WeightCost
        {
            get { return Dispatcher.InvokeIfRequired(() => (double)GetValue(WeightCostProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(WeightCostProperty, value)).Wait(); }
        }

        public double InitialMomentum
        {
            get { return Dispatcher.InvokeIfRequired(() => (double)GetValue(InitialMomentumProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(InitialMomentumProperty, value)).Wait(); }
        }

        public double FinalMomentum
        {
            get { return Dispatcher.InvokeIfRequired(() => (double)GetValue(FinalMomentumProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(FinalMomentumProperty, value)).Wait(); }
        }

        public double WeightInitializationStDev
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (double)GetValue(WeightInitializationStDevProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(WeightInitializationStDevProperty, value)).Wait(); }
        }
    }
}