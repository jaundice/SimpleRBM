using System.Windows;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;
#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;
#endif
namespace CudaNN.DeepBelief.LayerBuilders
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
            DependencyProperty.Register("WeightCost", typeof(TElement),
                typeof(ConstructNewLayer), new PropertyMetadata((TElement)2E-04));

        public static readonly DependencyProperty InitialMomentumProperty =
            DependencyProperty.Register("InitialMomentum", typeof(TElement),
                typeof(ConstructNewLayer), new PropertyMetadata((TElement)0.5));


        public static readonly DependencyProperty FinalMomentumProperty =
            DependencyProperty.Register("FinalMomentum", typeof(TElement),
                typeof(ConstructNewLayer), new PropertyMetadata((TElement)0.9));

        public static readonly DependencyProperty WeightInitializationStDevProperty =
            DependencyProperty.Register("WeightInitializationStDev", typeof(TElement),
                typeof(ConstructNewLayer), new PropertyMetadata((TElement)0.01));

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

        public TElement WeightCost
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement)GetValue(WeightCostProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(WeightCostProperty, value)).Wait(); }
        }

        public TElement InitialMomentum
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement)GetValue(InitialMomentumProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(InitialMomentumProperty, value)).Wait(); }
        }

        public TElement FinalMomentum
        {
            get { return Dispatcher.InvokeIfRequired(() => (TElement)GetValue(FinalMomentumProperty)).Result; }
            set { Dispatcher.InvokeIfRequired(() => SetValue(FinalMomentumProperty, value)).Wait(); }
        }

        public TElement WeightInitializationStDev
        {
            get
            {
                return Dispatcher.InvokeIfRequired(() => (TElement)GetValue(WeightInitializationStDevProperty)).Result;
            }
            set { Dispatcher.InvokeIfRequired(() => SetValue(WeightInitializationStDevProperty, value)).Wait(); }
        }
    }
}