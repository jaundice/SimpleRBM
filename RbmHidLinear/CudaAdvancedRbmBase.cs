using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Mono.CSharp;
using SimpleRBM.Common;
using SimpleRBM.Cuda;
#if USEFLOAT
using TElement = System.Single;

#else
using TElement = System.Double;

#endif

namespace CudaNN
{
    public abstract class CudaAdvancedRbmBase : IAdvancedRbmCuda<TElement>
    {
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
            int numHiddenNeurons, TElement weightcost = (TElement) 0.0002,
            TElement initialMomentum = (TElement) 0.5, TElement finalMomentum = (TElement) 0.9, TElement weightInitializationStDev = (TElement)0.01)
        {
            _weightcost = weightcost;
            _initialmomentum = initialMomentum;
            _finalmomentum = finalMomentum;
            _numHiddenNeurons = numHiddenNeurons;
            _numVisibleNeurons = numVisibleNeurons;
            _layerIndex = layerIndex;
            _gpu = gpu;
            _rand = rand;

            _weights = _gpu.GuassianDistribution(_rand, _numVisibleNeurons, _numHiddenNeurons,
                stDev: weightInitializationStDev);

            //https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

            _hiddenBiases = _gpu.AllocateAndSet<TElement>(1, _numHiddenNeurons);
            _visibleBiases = _gpu.AllocateAndSet<TElement>(1, _numVisibleNeurons);
            _vishidinc = _gpu.AllocateAndSet<TElement>(_numVisibleNeurons, _numHiddenNeurons);
            _visbiasinc = _gpu.AllocateAndSet<TElement>(1, _numVisibleNeurons);
            _hidbiasinc = _gpu.AllocateAndSet<TElement>(1, _numHiddenNeurons);

            Suspend();
        } /*
                      info.AddValue("numVisNeurons", NumVisibleNeurons);
            info.AddValue("numHidNeurons", NumHiddenNeurons);
            info.AddValue("weightCost", WeightCost);
            info.AddValue("initMomentum", InitialMomentum);
            info.AddValue("finalMomentum", FinalMomentum);
            info.AddValue("index", LayerIndex);
            info.AddValue("state", State, typeof(SuspendState));

            if (State == SuspendState.Suspended)
            {
                info.AddValue("hidBias", _cache[0], typeof(TElement[]));
                info.AddValue("visBias", _cache[1], typeof(TElement[]));
                info.AddValue("weights", _cache[2], typeof(TElement[]));
                info.AddValue("hidBiasInc", _cache[3], typeof(TElement[]));
                info.AddValue("visBiasInc", _cache[4], typeof(TElement[]));
                info.AddValue("weightInc", _cache[5], typeof(TElement[]));

            }
          * */

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

        public virtual void GreedyTrain(Matrix2D<TElement> data,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        {
            var state = State;
            Wake();
            int numcases = data.GetLength(0);
            exitConditionEvaluator.Start();
            var sw = new Stopwatch();
            using (Matrix2D<TElement> dataTransposed = data.Transpose())
            using (Matrix2D<TElement> posvisact = data.SumColumns())
            {
                int epoch;
                TElement error;
                EpochEventArgs<TElement> args;
                for (epoch = 0; ; epoch++)
                {
                    sw.Restart();
                    TElement weightLearningRate = weightLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                    TElement visBiasLearningRate = visBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                    TElement hidBiasLearningRate = hidBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);

                    error = BatchedTrainEpoch(data, dataTransposed, posvisact, epoch, numcases,
                        weightLearningRate, hidBiasLearningRate, visBiasLearningRate);

                    TElement delta;
                    var shouldExit = exitConditionEvaluator.Exit(epoch, error, sw.Elapsed, out delta);
                    args = new EpochEventArgs<TElement>()
                    {
                        Epoch = epoch, Error = error, Layer = LayerIndex, LearningRate = weightLearningRate, Elapsed = sw.Elapsed, Delta = delta
                    };
                    OnEpochComplete(args);
                    if (shouldExit)
                        break;
                }

                OnTrainComplete(args);
            }
            SetState(state);
        }

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


        public virtual void GreedyBatchedTrain(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        {
            var state = State;
            Wake();
            int numcases = data.GetLength(0);

            exitConditionEvaluator.Start();
            var datasets = PartitionDataAsMatrices(data, batchSize);
            try
            {
                Stopwatch sw = new Stopwatch();
                int epoch;
                TElement error;
                EpochEventArgs<TElement> args;
                for (epoch = 0; ; epoch++)
                {
                    sw.Restart();
                    TElement weightLearningRate = weightLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                    TElement visBiasLearningRate = visBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                    TElement hidBiasLearningRate = hidBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);

                    error =
                        datasets.Sum(block => BatchedTrainEpoch(block.Item1, block.Item2, block.Item3, epoch, numcases,
                            weightLearningRate, hidBiasLearningRate, visBiasLearningRate));



                    TElement delta;
                    var shouldExit = exitConditionEvaluator.Exit(epoch, error, sw.Elapsed, out delta);
                    args = new EpochEventArgs<TElement>()
                    {
                        Epoch = epoch,
                        Error = error,
                        Layer = LayerIndex,
                        LearningRate = weightLearningRate,
                        Elapsed = sw.Elapsed,
                        Delta = delta
                    };
                    OnEpochComplete(args);
                    if (shouldExit)
                        break;
                }

                OnTrainComplete(args);
            }
            finally
            {
                foreach (var dataset in datasets)
                {
                    dataset.Item1.Dispose();
                    dataset.Item2.Dispose();
                    dataset.Item3.Dispose();
                }
            }

            exitConditionEvaluator.Stop();
            SetState(state);
        }

        /// <summary>
        /// Same as GreedyBatchedTrain but the param data is disposed as soon as partitions are created to save gpu memory.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="batchSize"></param>
        /// <param name="exitConditionEvaluator"></param>
        /// <param name="weightLearningRateCalculator"></param>
        /// <param name="hidBiasLearningRateCalculator"></param>
        /// <param name="visBiasLearningRateCalculator"></param>
        public virtual void GreedyBatchedTrainMem(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        {
            var state = State;


            exitConditionEvaluator.Start();
            int numcases = data.GetLength(0);

            List<System.Tuple<TElement[,], TElement[,], TElement[,]>> datasets;

            Suspend(); //free memory for processing dataset
            using (data)
            {
                datasets = PartitionDataAsArrays(data, batchSize);
            }
            Wake();

            Stopwatch sw = new Stopwatch();
            int epoch;
            TElement error;
            EpochEventArgs<TElement> args;
            for (epoch = 0; ; epoch++)
            {
                sw.Restart();
                TElement weightLearningRate = weightLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                TElement visBiasLearningRate = visBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                TElement hidBiasLearningRate = hidBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);

                error = datasets.Sum(block =>
                {
                    using (var d = AsCuda.GPU.Upload(block.Item1))
                    using (var t = AsCuda.GPU.Upload(block.Item2))
                    using (var p = AsCuda.GPU.Upload(block.Item3))
                        return BatchedTrainEpoch(d, t, p, epoch, numcases,
                            weightLearningRate, hidBiasLearningRate,
                            visBiasLearningRate);
                });

                TElement delta;
                var shouldExit = exitConditionEvaluator.Exit(epoch, error, sw.Elapsed, out delta);
                args = new EpochEventArgs<TElement>()
                {
                    Epoch = epoch,
                    Error = error,
                    Layer = LayerIndex,
                    LearningRate = weightLearningRate,
                    Elapsed = sw.Elapsed,
                    Delta = delta
                };
                OnEpochComplete(args);
                if (shouldExit)
                    break;
            }

            OnTrainComplete(args);
            exitConditionEvaluator.Stop();
            SetState(state);
        }


        public SuspendState State { get; protected set; }

        public Matrix2D<TElement> HiddenBiasInc
        {
            get { return _hidbiasinc; }
        }

        public Matrix2D<TElement> VisibleBiasInc
        {
            get { return _visbiasinc; }
        }

        public Matrix2D<TElement> WeightInc
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

        protected abstract TElement BatchedTrainEpoch(Matrix2D<TElement> data, Matrix2D<TElement> dataTransposed,
            Matrix2D<TElement> posvisact,
            int epoch, int numcases, TElement weightLearningRate,
            TElement hidBiasLearningRate,
            TElement visBiasLearningRate);

        protected virtual List<System.Tuple<Matrix2D<TElement>, Matrix2D<TElement>, Matrix2D<TElement>>>
            PartitionDataAsMatrices(
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

        protected virtual List<System.Tuple<TElement[,], TElement[,], TElement[,]>> PartitionDataAsArrays(
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


        public virtual void Save(string path)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            var ss = new SurrogateSelector();
            var rbmSurrogate = new RbmSurrogate(_gpu, _rand);
            ss.AddSurrogate(typeof(CudaAdvancedRbmBinary), formatter.Context, rbmSurrogate);
            ss.AddSurrogate(typeof(CudaAdvancedRbmLinearHidden), formatter.Context, rbmSurrogate);
            formatter.SurrogateSelector = ss;
            using (var s = File.OpenWrite(path))
            {
                formatter.Serialize(s, this);
                s.Flush(true);
            }
        }

        public static IAdvancedRbmCuda<TElement> Deserialize(string path, GPGPU gpu, GPGPURAND rand)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            var ss = new SurrogateSelector();
            var rbmSurrogate = new RbmSurrogate(gpu, rand);
            ss.AddSurrogate(typeof(CudaAdvancedRbmBinary), formatter.Context, rbmSurrogate);
            ss.AddSurrogate(typeof(CudaAdvancedRbmLinearHidden), formatter.Context, rbmSurrogate);
            formatter.SurrogateSelector = ss;
            using (var s = File.OpenRead(path))
            {
                return (IAdvancedRbmCuda<TElement>)formatter.Deserialize(s);
            }
        }

        protected virtual void SaveSpecific(SerializationInfo info)
        {
        }

        protected virtual void LoadSpecific(SerializationInfo info)
        {
        }

        private class RbmSurrogate : ISerializationSurrogate
        {
            private GPGPU _gpu;
            private GPGPURAND _rand;

            public RbmSurrogate(GPGPU gpu, GPGPURAND rand)
            {
                this._gpu = gpu;
                this._rand = rand;
            }

            public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
            {
                var rbm = (CudaAdvancedRbmBase)obj;

                info.AddValue("numVisNeurons", rbm.NumVisibleNeurons);
                info.AddValue("numHidNeurons", rbm.NumHiddenNeurons);
                info.AddValue("weightCost", rbm.WeightCost);
                info.AddValue("initMomentum", rbm.InitialMomentum);
                info.AddValue("finalMomentum", rbm.FinalMomentum);
                info.AddValue("index", rbm.LayerIndex);
                info.AddValue("state", rbm.State, typeof(SuspendState));

                if (rbm.State == SuspendState.Suspended)
                {
                    info.AddValue("hidBias", rbm._cache[0], typeof(TElement[,]));
                    info.AddValue("visBias", rbm._cache[1], typeof(TElement[,]));
                    info.AddValue("weights", rbm._cache[2], typeof(TElement[,]));
                    info.AddValue("hidBiasInc", rbm._cache[3], typeof(TElement[,]));
                    info.AddValue("visBiasInc", rbm._cache[4], typeof(TElement[,]));
                    info.AddValue("weightInc", rbm._cache[5], typeof(TElement[,]));
                }
                else
                {
                    info.AddValue("hidBias", rbm._hiddenBiases.CopyLocal(), typeof(TElement[,]));
                    info.AddValue("visBias", rbm._visibleBiases.CopyLocal(), typeof(TElement[,]));
                    info.AddValue("weights", rbm._weights.CopyLocal(), typeof(TElement[,]));
                    info.AddValue("hidBiasInc", rbm._hidbiasinc.CopyLocal(), typeof(TElement[,]));
                    info.AddValue("visBiasInc", rbm._visbiasinc.CopyLocal(), typeof(TElement[,]));
                    info.AddValue("weightInc", rbm._vishidinc.CopyLocal(), typeof(TElement[,]));
                }

                rbm.SaveSpecific(info);
            }

            public object SetObjectData(object obj, SerializationInfo info, StreamingContext context,
                ISurrogateSelector selector)
            {
                var rbm = (CudaAdvancedRbmBase)obj;
                rbm._gpu = _gpu;
                rbm._rand = _rand;
                rbm._numVisibleNeurons = info.GetInt32("numVisNeurons");
                rbm._numHiddenNeurons = info.GetInt32("numHidNeurons");
                rbm._weightcost = (TElement)info.GetValue("weightCost", typeof(TElement));
                rbm._initialmomentum = (TElement)info.GetValue("initMomentum", typeof(TElement));
                rbm._finalmomentum = (TElement)info.GetValue("finalMomentum", typeof(TElement));
                rbm._layerIndex = info.GetInt32("index");
                var state = (SuspendState)info.GetValue("state", typeof(SuspendState));
                rbm._hiddenBiases = _gpu.Upload((TElement[,])info.GetValue("hidBias", typeof(TElement[,])));
                rbm._visibleBiases = _gpu.Upload((TElement[,])info.GetValue("visBias", typeof(TElement[,])));
                rbm._weights = _gpu.Upload((TElement[,])info.GetValue("weights", typeof(TElement[,])));
                rbm._hidbiasinc = _gpu.Upload((TElement[,])info.GetValue("hidBiasInc", typeof(TElement[,])));
                rbm._visbiasinc = _gpu.Upload((TElement[,])info.GetValue("visBiasInc", typeof(TElement[,])));
                rbm._vishidinc = _gpu.Upload((TElement[,])info.GetValue("weightInc", typeof(TElement[,])));

                rbm.LoadSpecific(info);

                rbm.SetState(state);

                return obj;
            }
        }


        public virtual void GreedyBatchedTrainMem(IList<TElement[,]> batches,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator)
        {
            var state = State;


            exitConditionEvaluator.Start();
            int numcases = batches.Sum(a => a.GetLength(0));
            Suspend();
            var datasets = new List<System.Tuple<TElement[,], TElement[,], TElement[,]>>();
            foreach (var batch in batches)
            {
                using (var d = _gpu.Upload(batch))
                using (var t = d.Transpose())
                using (var s = d.SumColumns())
                {
                    datasets.Add(Tuple.Create(batch, t.CopyLocal(), s.CopyLocal()));
                }
            }


            Wake();

            Stopwatch sw = new Stopwatch();
            int epoch;
            TElement error;
            EpochEventArgs<TElement> args;
            for (epoch = 0; ; epoch++)
            {
                sw.Restart();
                TElement weightLearningRate = weightLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                TElement visBiasLearningRate = visBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);
                TElement hidBiasLearningRate = hidBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch);

                error = datasets.Sum(block =>
                {
                    using (var d = AsCuda.GPU.Upload(block.Item1))
                    using (var t = AsCuda.GPU.Upload(block.Item2))
                    using (var p = AsCuda.GPU.Upload(block.Item3))
                        return BatchedTrainEpoch(d, t, p, epoch, numcases,
                            weightLearningRate, hidBiasLearningRate,
                            visBiasLearningRate);
                });

                TElement delta;
                var shouldExit = exitConditionEvaluator.Exit(epoch, error, sw.Elapsed, out delta);
                args = new EpochEventArgs<TElement>()
                {
                    Epoch = epoch,
                    Error = error,
                    Layer = LayerIndex,
                    LearningRate = weightLearningRate,
                    Elapsed = sw.Elapsed,
                    Delta = delta
                };
                OnEpochComplete(args);
                if (shouldExit)
                    break;
            }

            OnTrainComplete(args);
            exitConditionEvaluator.Stop();
            SetState(state);
        }
    }
}