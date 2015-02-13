using System;
using System.Diagnostics;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Mono.CSharp;
using SimpleRBM.Common;
using SimpleRBM.Cuda;
#if USEFLOAT
using TElementType = System.Single;
#else
using TElementType = System.Double;
#endif
namespace CudaNN
{
    public abstract class CudaAdvancedRbmBase : IDisposable, IAdvancedRbmCuda<TElementType>
    {
        //private TElementType _epsilonhb;
        //private TElementType _epsilonvb;
        //private TElementType _epsilonw;
        private TElementType _finalmomentum;
        private GPGPU _gpu;
        private Matrix2D<TElementType> _hiddenBiases;
        private TElementType _initialmomentum;
        private int _numHiddenNeurons;
        private GPGPURAND _rand;
        private Matrix2D<TElementType> _visibleBiases;
        private TElementType _weightcost;
        private Matrix2D<TElementType> _weights;
        protected Matrix2D<TElementType> _hidbiasinc;
        private int _numVisibleNeurons;
        protected Matrix2D<TElementType> _visbiasinc;
        protected Matrix2D<TElementType> _vishidinc;
        private int _layerIndex;

        public event EventHandler<EpochEventArgs<TElementType>> EpochEnd;
        public event EventHandler<EpochEventArgs<TElementType>> TrainEnd;

        protected void OnEpochComplete(EpochEventArgs<TElementType> args)
        {
            if (EpochEnd != null)
            {
                EpochEnd(this, args);
            }
        }

        protected void OnTrainComplete(EpochEventArgs<TElementType> args)
        {
            if (TrainEnd != null)
            {
                TrainEnd(this, args);
            }
        }
        protected CudaAdvancedRbmBase(GPGPU gpu, GPGPURAND rand, int layerIndex, int numVisibleNeurons, int numHiddenNeurons,
            /*TElementType epsilonw = (TElementType) 0.001, TElementType epsilonvb = (TElementType) 0.001,
            TElementType epsilonhb = (TElementType) 0.001,*/ TElementType weightcost = (TElementType) 0.0002,
            TElementType initialMomentum = (TElementType) 0.5, TElementType finalMomentum = (TElementType) 0.9)
        {
            //_epsilonw = epsilonw; // Learning rate for weights 
            //_epsilonvb = epsilonvb; // Learning rate for biases of visible units
            //_epsilonhb = epsilonhb; // Learning rate for biases of hidden units 
            _weightcost = weightcost;
            _initialmomentum = initialMomentum;
            _finalmomentum = finalMomentum;
            _numHiddenNeurons = numHiddenNeurons;
            _numVisibleNeurons = numVisibleNeurons;
            _layerIndex = layerIndex;
            _gpu = gpu;
            _rand = rand;

            _weights = _gpu.GuassianDistribution( _rand, _numVisibleNeurons, _numHiddenNeurons,
                scale: (TElementType)0.1);
            _hiddenBiases = _gpu.AllocateAndSet<TElementType>(1, _numHiddenNeurons);
            _visibleBiases = _gpu.AllocateAndSet<TElementType>(1, _numVisibleNeurons);
            _vishidinc = _gpu.AllocateAndSet<TElementType>(_numVisibleNeurons, _numHiddenNeurons);
            _visbiasinc = _gpu.AllocateAndSet<TElementType>(1, _numVisibleNeurons);
            _hidbiasinc = _gpu.AllocateAndSet<TElementType>(1, _numHiddenNeurons);
        }

        public bool Disposed { get; protected set; }

         Matrix2D<TElementType> IAdvancedRbmCuda<TElementType>.HiddenBiases
        {
            get { return _hiddenBiases; }
        }

         Matrix2D<TElementType> IAdvancedRbmCuda<TElementType>.VisibleBiases
        {
            get { return _visibleBiases; }
        }

         Matrix2D<TElementType> IAdvancedRbmCuda<TElementType>.Weights
        {
            get { return _weights; }
        }

        //public TElementType EpsilonHiddenBias
        //{
        //    get { return _epsilonhb; }
        //}

        //public TElementType EpsilonVisibleBias
        //{
        //    get { return _epsilonvb; }
        //}

        //public TElementType EpsilonWeight
        //{
        //    get { return _epsilonw; }
        //}

        public TElementType FinalMomentum
        {
            get { return _finalmomentum; }
        }

        public TElementType InitialMomentum
        {
            get { return _initialmomentum; }
        }

        public TElementType WeightCost
        {
            get { return _weightcost; }
        }

        GPGPU IAdvancedRbmCuda<TElementType>.GPU
        {
            get { return _gpu; }
        }

        public int LayerIndex
        {
            get { return _layerIndex; }
        }

        GPGPURAND IAdvancedRbmCuda<TElementType>.GPURAND
        {
            get { return _rand; }
        }

        public int NumVisibleNeurons
        {
            get { return _numVisibleNeurons; }
        }

        public int NumHiddenNeurons
        {
            get { return _numHiddenNeurons; }
        }

        public void Dispose()
        {
            if (!Disposed)
            {
                Disposed = true;
                Dispose(true);
                GC.SuppressFinalize(this);
            }
        }

        public void GreedyTrain(TElementType[,] visibleData, IExitConditionEvaluator<TElementType> exitConditionEvaluator, ILearningRateCalculator<TElementType> weightLearningRateCalculator, ILearningRateCalculator<TElementType> hidBiasLearningRateCalculator, ILearningRateCalculator<TElementType> visBiasLearningRateCalculator)
        {
            using (Matrix2D<TElementType> data = _gpu.Upload(visibleData))
            {
                GreedyTrain(data, exitConditionEvaluator, weightLearningRateCalculator, hidBiasLearningRateCalculator, visBiasLearningRateCalculator);
            }
        }

        public TElementType[,] Encode(TElementType[,] srcData)
        {
            using (var data = _gpu.Upload(srcData))
            using (var res = Encode(data))
            {
                return res.CopyLocal();
            }
        }

        public abstract Matrix2D<TElementType> Encode(Matrix2D<TElementType> data);

        public TElementType[,] Decode(TElementType[,] activations)
        {
            using (var act = _gpu.Upload(activations))
            using (var res = Decode(act))
            {
                return res.CopyLocal();
            }
        }

        public abstract Matrix2D<TElementType> Decode(Matrix2D<TElementType> activations);

        public TElementType[,] Reconstruct(TElementType[,] data)
        {
            using(var d = _gpu.Upload(data))
            using (var res = Reconstruct(d))
                return res.CopyLocal();
        }

        public virtual Matrix2D<TElementType> Reconstruct(Matrix2D<TElementType> data)
        {
            using (var res = Encode(data))
                return Decode(res);
        }
        
        public abstract void GreedyTrain(Matrix2D<TElementType> data, IExitConditionEvaluator<TElementType> exitConditionEvaluator, ILearningRateCalculator<TElementType> weightLearningRateCalculator, ILearningRateCalculator<TElementType> hidBiasLearningRateCalculator, ILearningRateCalculator<TElementType> visBiasLearningRateCalculator   );

        public void Dispose(bool disposing)
        {
            if (disposing)
            {
               

                AsCuda.Weights.Dispose();
                AsCuda.HiddenBiases.Dispose();
                AsCuda.VisibleBiases.Dispose();
                _vishidinc.Dispose();
                _visbiasinc.Dispose();
                _hidbiasinc.Dispose();
            }
        }

        protected IAdvancedRbmCuda<TElementType> AsCuda
        {
            get { return this; }
        }

        ~CudaAdvancedRbmBase()
        {
            Trace.TraceError("Finalizer called. Dispose of properly");
            Dispose(false);
        }


        public TElementType[,] DayDream(int numberOfSamples)
        {
            throw new NotImplementedException();
        }

        public void GreedyTrain(TElementType[,] visibleData, IExitConditionEvaluator<TElementType> exitEvaluator, ILearningRateCalculator<TElementType> learningRateCalculator)
        {
            GreedyTrain(visibleData, exitEvaluator, learningRateCalculator, learningRateCalculator,
                learningRateCalculator);
        }

        public ILayerSaveInfo<TElementType> GetSaveInfo()
        {
            throw new NotImplementedException();
        }

        public TElementType CalculateReconstructionError(TElementType[,] data)
        {
            throw new NotImplementedException();
        }
    }
}