using System;
using System.Collections.Generic;
using System.Diagnostics;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Mono.CSharp;
using SimpleRBM.Common;
using SimpleRBM.Cuda;
#if USEFLOAT
using TElement = System.Single;

#else
using TElementType = System.Double;
#endif

namespace CudaNN
{
    public abstract class CudaAdvancedRbmBase : IDisposable, IAdvancedRbmCuda<TElement>
    {
        //private TElementType _epsilonhb;
        //private TElementType _epsilonvb;
        //private TElementType _epsilonw;
        private TElement _finalmomentum;
        private GPGPU _gpu;
        private int _layerIndex;
        private TElement _initialmomentum;
        private int _numVisibleNeurons;
        private int _numHiddenNeurons;
        private GPGPURAND _rand;
        private TElement _weightcost;

        protected Matrix2D<TElement> _hiddenBiases;
        protected Matrix2D<TElement> _visibleBiases;
        protected Matrix2D<TElement> _weights;
        protected Matrix2D<TElement> _hidbiasinc;
        protected Matrix2D<TElement> _visbiasinc;
        protected Matrix2D<TElement> _vishidinc;

        public event EventHandler<EpochEventArgs<TElement>> EpochEnd;
        public event EventHandler<EpochEventArgs<TElement>> TrainEnd;

        protected void OnEpochComplete(EpochEventArgs<TElement> args)
        {
            if (EpochEnd != null)
            {
                EpochEnd(this, args);
            }
        }

        protected void OnTrainComplete(EpochEventArgs<TElement> args)
        {
            if (TrainEnd != null)
            {
                TrainEnd(this, args);
            }
        }

        protected CudaAdvancedRbmBase(GPGPU gpu, GPGPURAND rand, int layerIndex, int numVisibleNeurons,
            int numHiddenNeurons,
            /*TElementType epsilonw = (TElementType) 0.001, TElementType epsilonvb = (TElementType) 0.001,
            TElementType epsilonhb = (TElementType) 0.001,*/ TElement weightcost = (TElement) 0.0002,
            TElement initialMomentum = (TElement) 0.5, TElement finalMomentum = (TElement) 0.9)
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

            _weights = _gpu.GuassianDistribution(_rand, _numVisibleNeurons, _numHiddenNeurons,
                scale: (TElement)0.1);
            _hiddenBiases = _gpu.AllocateAndSet<TElement>(1, _numHiddenNeurons);
            _visibleBiases = _gpu.AllocateAndSet<TElement>(1, _numVisibleNeurons);
            _vishidinc = _gpu.AllocateAndSet<TElement>(_numVisibleNeurons, _numHiddenNeurons);
            _visbiasinc = _gpu.AllocateAndSet<TElement>(1, _numVisibleNeurons);
            _hidbiasinc = _gpu.AllocateAndSet<TElement>(1, _numHiddenNeurons);

            Suspend();
        }

        public bool Disposed { get; protected set; }

        Matrix2D<TElement> IAdvancedRbmCuda<TElement>.HiddenBiases
        {
            get { return _hiddenBiases; }
        }

        Matrix2D<TElement> IAdvancedRbmCuda<TElement>.VisibleBiases
        {
            get { return _visibleBiases; }
        }

        Matrix2D<TElement> IAdvancedRbmCuda<TElement>.Weights
        {
            get { return _weights; }
        }


        public TElement FinalMomentum
        {
            get { return _finalmomentum; }
        }

        public TElement InitialMomentum
        {
            get { return _initialmomentum; }
        }

        public TElement WeightCost
        {
            get { return _weightcost; }
        }

        GPGPU IAdvancedRbmCuda<TElement>.GPU
        {
            get { return _gpu; }
        }

        public int LayerIndex
        {
            get { return _layerIndex; }
        }

        GPGPURAND IAdvancedRbmCuda<TElement>.GPURAND
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

        public void GreedyTrain(TElement[,] visibleData,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        {
            using (Matrix2D<TElement> data = _gpu.Upload(visibleData))
            {
                GreedyTrain(data, exitConditionEvaluator, weightLearningRateCalculator, hidBiasLearningRateCalculator,
                    visBiasLearningRateCalculator);
            }
        }

        public TElement[,] Encode(TElement[,] srcData)
        {
            using (var data = _gpu.Upload(srcData))
            using (var res = Encode(data))
            {
                return res.CopyLocal();
            }
        }

        public abstract Matrix2D<TElement> Encode(Matrix2D<TElement> data);

        public TElement[,] Decode(TElement[,] activations)
        {
            using (var act = _gpu.Upload(activations))
            using (var res = Decode(act))
            {
                return res.CopyLocal();
            }
        }

        public abstract Matrix2D<TElement> Decode(Matrix2D<TElement> activations);

        public TElement[,] Reconstruct(TElement[,] data)
        {
            using (var d = _gpu.Upload(data))
            using (var res = Reconstruct(d))
                return res.CopyLocal();
        }

        public virtual Matrix2D<TElement> Reconstruct(Matrix2D<TElement> data)
        {
            using (var res = Encode(data))
                return Decode(res);
        }

        public abstract void GreedyTrain(Matrix2D<TElement> data,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator);

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

        protected IAdvancedRbmCuda<TElement> AsCuda
        {
            get { return this; }
        }

        ~CudaAdvancedRbmBase()
        {
            Trace.TraceError("Finalizer called. Dispose of properly");
            Dispose(false);
        }


        public TElement[,] DayDream(int numberOfSamples)
        {
            throw new NotImplementedException();
        }

        public void GreedyTrain(TElement[,] visibleData, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            GreedyTrain(visibleData, exitEvaluator, learningRateCalculator, learningRateCalculator,
                learningRateCalculator);
        }

        public ILayerSaveInfo<TElement> GetSaveInfo()
        {
            throw new NotImplementedException();
        }

        public TElement CalculateReconstructionError(TElement[,] data)
        {
            throw new NotImplementedException();
        }


        public abstract void GreedyBatchedTrain(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator);

        /// <summary>
        /// Same as GreedyBatchedTrain but the param data is disposed as soon as partitions are created to save gpu memory.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="batchSize"></param>
        /// <param name="exitConditionEvaluator"></param>
        /// <param name="weightLearningRateCalculator"></param>
        /// <param name="hidBiasLearningRateCalculator"></param>
        /// <param name="visBiasLearningRateCalculator"></param>
        public abstract void GreedyBatchedTrainMem(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator);


        public SuspendState State { get; protected set; }

        public Matrix2D<float> HiddenBiasInc
        {
            get { return _hidbiasinc; }
        }

        public Matrix2D<float> VisibleBiasInc
        {
            get { return _visbiasinc; }
        }

        public Matrix2D<float> WeightInc
        {
            get { return _vishidinc; }
        }

        public void Suspend()
        {
            if (State != SuspendState.Suspended)
            {
                DoSuspend();
                State = SuspendState.Suspended;
            }
        }


        public void Wake()
        {
            if (State != SuspendState.Active)
            {
                DoWake();
                State = SuspendState.Active;
            }
        }

        private List<TElement[,]> _cache;


        protected virtual void DoSuspend()
        {
            _cache = new List<TElement[,]>
            {
                _hiddenBiases.CopyLocal(),
                _visibleBiases.CopyLocal(),
                _weights.CopyLocal(),
                _hidbiasinc.CopyLocal(),
                _visbiasinc.CopyLocal(),
                _vishidinc.CopyLocal()
            };

            _hiddenBiases.Dispose();
            _visibleBiases.Dispose();
            _weights.Dispose();
            _hidbiasinc.Dispose();
            _visbiasinc.Dispose();
            _vishidinc.Dispose();
        }

        protected virtual void DoWake()
        {
            _hiddenBiases = AsCuda.GPU.Upload(_cache[0]);
            _visibleBiases = AsCuda.GPU.Upload(_cache[1]);
            _weights = AsCuda.GPU.Upload(_cache[2]);
            _hidbiasinc = AsCuda.GPU.Upload(_cache[3]);
            _visbiasinc = AsCuda.GPU.Upload(_cache[4]);
            _vishidinc = AsCuda.GPU.Upload(_cache[5]);

            _cache.Clear();
            _cache = null;
        }

        public void SetState(SuspendState state)
        {
            switch (state)
            {
                case SuspendState.Active:
                    Wake();
                    break;
                case SuspendState.Suspended:
                    Suspend();
                    break;
            }
        }

        protected virtual List<System.Tuple<Matrix2D<TElement>, Matrix2D<TElement>, Matrix2D<TElement>>> PartitionDataAsMatrices(
            Matrix2D<TElement> data, int batchSize)
        {
            var datasets = new List<System.Tuple<Matrix2D<TElement>, Matrix2D<TElement>, Matrix2D<TElement>>>();
            for (int j = 0; j < data.GetLength(0); j += batchSize)
            {
                int endIndex = j + batchSize;
                if (endIndex > data.GetLength(0) - 1)
                    endIndex = data.GetLength(0) - 1;

                int examples = endIndex - j;
                Matrix2D<TElement> part = data.SubMatrix(j, 0, examples);
                Matrix2D<TElement> trans = part.Transpose();
                Matrix2D<TElement> posVisAct = part.SumColumns();

                datasets.Add(Tuple.Create(part, trans, posVisAct));
            }

            return datasets;
        }

        protected  virtual List<System.Tuple<TElement[,], TElement[,], TElement[,]>> PartitionDataAsArrays(
            Matrix2D<TElement> data, int batchSize)
        {
            var datasets = new List<System.Tuple<TElement[,], TElement[,], TElement[,]>>();
            for (int j = 0; j < data.GetLength(0); j += batchSize)
            {
                int endIndex = j + batchSize;
                if (endIndex > data.GetLength(0) - 1)
                    endIndex = data.GetLength(0) - 1;

                int examples = endIndex - j;

                using (Matrix2D<TElement> part = data.SubMatrix(j, 0, examples))
                using (Matrix2D<TElement> trans = part.Transpose())
                using (Matrix2D<TElement> posVisAct = part.SumColumns())
                {
                    datasets.Add(Tuple.Create(part.CopyLocal(), trans.CopyLocal(), posVisAct.CopyLocal()));
                }
            }

            return datasets;
        }
    }
}