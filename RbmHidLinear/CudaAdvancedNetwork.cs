using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Cuda;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;

#endif

namespace CudaNN
{
    public class CudaAdvancedNetwork : ICudaNetwork<TElement>
    {
        public CudaAdvancedNetwork(GPGPU gpu, GPGPURAND rand, string[] machineDataPaths)
            : this(machineDataPaths.Select(a => CudaAdvancedRbmBase.Deserialize(a, gpu, rand))
                .OrderBy(b => b.LayerIndex)
                .ToList())
        {
        }

        public CudaAdvancedNetwork(IEnumerable<IAdvancedRbmCuda<TElement>> machines)
        {
            Machines = machines.ToList();

            foreach (var machine in Machines)
            {
                machine.EpochEnd += (a, b) => OnEpochComplete(b);
                machine.TrainEnd += (a, b) => OnLayerTrainComplete(b);
            }
        }

        private bool Disposed { get; set; }

        protected ICudaNetwork<TElement> AsCuda
        {
            get { return this; }
        }

        public IList<IAdvancedRbmCuda<TElement>> Machines { get; protected set; }
        public event EventHandler<EpochEventArgs<TElement>> EpochComplete;
        public event EventHandler<EpochEventArgs<TElement>> LayerTrainComplete;


        public TElement[,] Reconstruct(TElement[,] data, int maxDepth = -1)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            using (Matrix2D<TElement> res = AsCuda.Reconstruct(d, maxDepth))
                return res.CopyLocal();
        }

        public TElement[,] Encode(TElement[,] data, int maxDepth = -1)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            using (Matrix2D<TElement> res = AsCuda.Encode(d, maxDepth))
                return res.CopyLocal();
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.Encode(Matrix2D<TElement> data, int maxDepth = -1)
        {
            Matrix2D<TElement> d = data;

            int dataLen = data.GetLength(0);


            int depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;
            for (int i = 0; i < depth + 1; i++)
            {


                Matrix2D<TElement> encoded = Machines[i].Encode(d);
                if (!ReferenceEquals(d, data))
                {
                    d.Dispose();
                }
                d = encoded;

                if (d.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
            return d;
        }

        /// <summary>
        ///     Same as Encode except the visible data is extended for the inner most rbm to the size of the visible buffer (i.e
        ///     numlabels columns are added)
        /// </summary>
        /// <param name="data"></param>
        /// <param name="maxDepth"></param>
        /// <returns></returns>
        Matrix2D<TElement> ICudaNetwork<TElement>.EncodeWithLabelExpansion(Matrix2D<TElement> data)
        {
            Matrix2D<TElement> d = data;
            int dataLen = data.GetLength(0);

            int depth = Machines.Count - 1;
            for (int i = 0; i < depth + 1; i++)
            {


                if (i == Machines.Count - 1)
                {
                    Matrix2D<TElement> expanded = Machines[0].GPU.AllocateAndSet<TElement>(data.GetLength(0),
                        Machines[i].NumVisibleNeurons);
                    expanded.InsertValuesFrom(0, 0, d);

                    if (!ReferenceEquals(d, data))
                        d.Dispose();

                    d = expanded;
                }

                Matrix2D<TElement> encoded = Machines[i].Encode(d);
                if (!ReferenceEquals(d, data))
                {
                    d.Dispose();
                }

                d = encoded;

                if (d.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
            return d;
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.Reconstruct(Matrix2D<TElement> data, int maxDepth = -1)
        {
            int depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;

            using (Matrix2D<TElement> encoded = AsCuda.Encode(data, depth))
                return AsCuda.Decode(encoded, depth);
        }


        public TElement[,] Decode(TElement[,] activations, int maxDepth = -1)
        {
            using (Matrix2D<TElement> act = Machines[0].GPU.Upload(activations))
            {
                using (Matrix2D<TElement> res = AsCuda.Decode(act, maxDepth))
                {
                    return res.CopyLocal();
                }
            }
        }


        Matrix2D<TElement> ICudaNetwork<TElement>.Decode(Matrix2D<TElement> activations, int maxDepth = -1)
        {
            int depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;

            int dataLen = activations.GetLength(0);
            Matrix2D<TElement> d = activations;

            for (int i = depth; i > -1; i--)
            {


                Matrix2D<TElement> constructed = Machines[i].Decode(d);
                if (!ReferenceEquals(d, activations))
                    d.Dispose();
                d = constructed;

                if (d.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }



            return d;
        }

        public TElement[,] ReconstructWithLabels(TElement[,] data,
            out TElement[,] labels, bool softmaxLabels = true)
        {
            Matrix2D<TElement> lbl;
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            using (Matrix2D<TElement> recon = AsCuda.ReconstructWithLabels(d, out lbl, softmaxLabels))
            using (lbl)
            {
                labels = lbl.CopyLocal();
                return recon.CopyLocal();
            }
        }


        Matrix2D<TElement> ICudaNetwork<TElement>.ReconstructWithLabels(Matrix2D<TElement> data,
            out Matrix2D<TElement> labels, bool softmaxLabels = true)
        {
            int depth = Machines.Count - 1;

            using (Matrix2D<TElement> d = AsCuda.EncodeWithLabelExpansion(data))
                return AsCuda.DecodeWithLabels(d, out labels, softmaxLabels);
        }

        public TElement[,] DecodeWithLabels(TElement[,] activations,
            out TElement[,] labels, bool softmaxLabels = true)
        {
            using (Matrix2D<TElement> act = Machines[0].GPU.Upload(activations))
            {
                Matrix2D<TElement> lbl;
                using (Matrix2D<TElement> res = AsCuda.DecodeWithLabels(act, out lbl, softmaxLabels))
                using (lbl)
                {
                    labels = lbl.CopyLocal();
                    return res.CopyLocal();
                }
            }
        }


        Matrix2D<TElement> ICudaNetwork<TElement>.DecodeWithLabels(Matrix2D<TElement> activations,
            out Matrix2D<TElement> labels, bool softmaxLabels = true)
        {
            int depth = Machines.Count - 1;
            int dataLen = activations.GetLength(0);

            labels = null;
            Matrix2D<TElement> d = activations;
            for (int i = depth; i > -1; i--)
            {


                Matrix2D<TElement> constructed = Machines[i].Decode(d);

                if (i == depth)
                {
                    labels = constructed.SubMatrix(0, Machines[i - 1].NumHiddenNeurons);
                    if (softmaxLabels)
                    {
                        using (labels)
                        {
                            Matrix2D<TElement> sm = labels.SoftMax();
                            //sm.ToBinary();
                            labels = sm;
                        }
                    }
                    Matrix2D<TElement> c = constructed.SubMatrix(0, 0, 0, Machines[i - 2].NumHiddenNeurons);
                    constructed.Dispose();
                    constructed = c;
                }
                if (!ReferenceEquals(d, activations))
                    d.Dispose();
                d = constructed;

                if (d.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
            return d;
        }


        public TElement[,] LabelData(TElement[,] data, bool softmaxLabels = true)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            using (Matrix2D<TElement> r = AsCuda.LabelData(d, softmaxLabels))
                return r.CopyLocal();
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.LabelData(Matrix2D<TElement> data, bool softmaxLabels = true)
        {
            int depth = Machines.Count - 1;
            using (Matrix2D<TElement> d = AsCuda.EncodeWithLabelExpansion(data))
            using (Matrix2D<TElement> constructed = Machines[depth].Decode(d))
            {
                Matrix2D<TElement> ret = constructed.SubMatrix(0, Machines[depth - 1].NumHiddenNeurons);
                if (softmaxLabels)
                {
                    using (ret)
                    {
                        Matrix2D<TElement> sm = ret.SoftMax();
                        //sm.ToBinary();
                        return sm;
                    }
                }
                return ret;
            }
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.Daydream(TElement noiseScale, int numDreams, int maxDepth = -1,
            bool guassian = true)
        {
            using (
                Matrix2D<TElement> rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, noiseScale)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, noiseScale))
            {
                return AsCuda.Reconstruct(rand, maxDepth);
            }
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.DaydreamWithLabels(TElement noiseScale, int numDreams,
            out Matrix2D<TElement> labels, bool guassian = true, bool softmaxLabels = true)
        {
            using (
                Matrix2D<TElement> rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, noiseScale)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, noiseScale))
            {
                return AsCuda.ReconstructWithLabels(rand, out labels, softmaxLabels);
            }
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.DaydreamByClass(TElement noiseScale, Matrix2D<TElement> modelLabels,
            out Matrix2D<TElement> generatedLabels, bool guassian = true, bool softmaxLabels = true)
        {
            int highest = Machines.Count - 1;
            using (
                Matrix2D<TElement> rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, scale: noiseScale)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, noiseScale))
            {
                rand.InsertValuesFrom(0, Machines[highest - 1].NumHiddenNeurons, modelLabels);
                using (Matrix2D<TElement> encoded = Machines[highest].Encode(rand))
                {
                    return AsCuda.DecodeWithLabels(encoded, out generatedLabels, softmaxLabels);
                }
            }
        }


        void ICudaNetwork<TElement>.GreedyTrain(Matrix2D<TElement> data,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken,
            int trainFrom = 0)
        {
            Matrix2D<TElement> layerTrainData = data;

            int dataLen = data.GetLength(0);

            for (int i = 0; i < Machines.Count; i++)
            {


                cancelToken.ThrowIfCancellationRequested();
                if (i >= trainFrom)
                    Machines[i].GreedyTrain(layerTrainData, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);
                Matrix2D<TElement> encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;

                if (layerTrainData.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
        }

        void ICudaNetwork<TElement>.GreedySupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken,
            int trainFrom = 0)
        {
            Matrix2D<TElement> layerTrainData = data;

            int dataLen = data.GetLength(0);

            for (int i = 0; i < Machines.Count; i++)
            {

                cancelToken.ThrowIfCancellationRequested();
                if (i == Machines.Count - 1)
                {
                    Matrix2D<TElement> combined = Machines[0].GPU.AllocateNoSet<TElement>(data.GetLength(0),
                        layerTrainData.GetLength(1) + labels.GetLength(1));

                    combined.InsertValuesFrom(0, 0, layerTrainData);
                    combined.InsertValuesFrom(0, layerTrainData.GetLength(1), labels);
                    layerTrainData.Dispose();
                    layerTrainData = combined;
                }

                if (i >= trainFrom)
                    Machines[i].GreedyTrain(layerTrainData, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);


                Matrix2D<TElement> encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;

                if (layerTrainData.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }

            if (!ReferenceEquals(layerTrainData, data))
            {
                layerTrainData.Dispose();
            }
        }

        void IDisposable.Dispose()
        {
            if (!Disposed)
            {
                Disposed = true;
                Dispose(true);
                GC.SuppressFinalize(this);
            }
        }

        IList<IRestrictedBoltzmannMachine<TElement>> INetwork<TElement>.Machines
        {
            get { return (IList<IRestrictedBoltzmannMachine<TElement>>)Machines; }
        }


        void ICudaNetwork<TElement>.GreedyBatchedTrain(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken,
            int trainFrom = 0)
        {
            Matrix2D<TElement> layerTrainData = data;
            int dataLen = data.GetLength(0);

            for (int i = 0; i < Machines.Count; i++)
            {
                cancelToken.ThrowIfCancellationRequested();
                if (i >= trainFrom)
                    Machines[i].GreedyBatchedTrain(layerTrainData, batchSize, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);
                Matrix2D<TElement> encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;

                if (layerTrainData.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
        }

        void ICudaNetwork<TElement>.GreedyBatchedSupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            int batchSize, IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken,
            int trainFrom = 0)
        {
            Matrix2D<TElement> layerTrainData = data;
            int dataLen = data.GetLength(0);

            for (int i = 0; i < Machines.Count; i++)
            {

                cancelToken.ThrowIfCancellationRequested();
                if (i == Machines.Count - 1)
                {
                    Matrix2D<TElement> combined = Machines[0].GPU.AllocateNoSet<TElement>(data.GetLength(0),
                        layerTrainData.GetLength(1) + labels.GetLength(1));

                    combined.InsertValuesFrom(0, 0, layerTrainData);
                    combined.InsertValuesFrom(0, layerTrainData.GetLength(1), labels);
                    layerTrainData.Dispose();
                    layerTrainData = combined;
                }

                if (i >= trainFrom)
                    Machines[i].GreedyBatchedTrain(layerTrainData, batchSize, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);


                Matrix2D<TElement> encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;

                if (layerTrainData.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }

            }

            if (!ReferenceEquals(layerTrainData, data))
            {
                layerTrainData.Dispose();
            }
        }


        void ICudaNetwork<TElement>.GreedyBatchedTrainMem(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken,
            int trainFrom = 0)
        {

            Matrix2D<TElement> layerTrainData = data;
            int dataLen = data.GetLength(0);

            for (int i = 0; i < Machines.Count; i++)
            {
                cancelToken.ThrowIfCancellationRequested();
                TElement[,] localCopy = layerTrainData.CopyLocal();
                if (i >= trainFrom)
                    Machines[i].GreedyBatchedTrainMem(layerTrainData, batchSize, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);

                Matrix2D<TElement> encoded;
                using (Matrix2D<TElement> up = Machines[0].GPU.Upload(localCopy))
                {
                    encoded = Machines[i].Encode(up);
                }

                layerTrainData = encoded;

                if (layerTrainData.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
        }

        void ICudaNetwork<TElement>.GreedyBatchedSupervisedTrainMem(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            int batchSize, IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken,
            int trainFrom = 0)
        {
            Matrix2D<TElement> layerTrainData = data;

            int dataLen = data.GetLength(0);

            for (int i = 0; i < Machines.Count; i++)
            {
                cancelToken.ThrowIfCancellationRequested();
                if (i == Machines.Count - 1)
                {
                    Matrix2D<TElement> combined = Machines[0].GPU.AllocateNoSet<TElement>(data.GetLength(0),
                        layerTrainData.GetLength(1) + labels.GetLength(1));

                    combined.InsertValuesFrom(0, 0, layerTrainData);
                    combined.InsertValuesFrom(0, layerTrainData.GetLength(1), labels);
                    labels.Dispose();
                    layerTrainData.Dispose();
                    layerTrainData = combined;
                }

                TElement[,] local = layerTrainData.CopyLocal();

                if (i >= trainFrom)
                {
                    Machines[i].GreedyBatchedTrainMem(layerTrainData, batchSize, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);
                }
                else
                {
                    layerTrainData.Dispose();
                }

                Matrix2D<TElement> encoded;
                using (Matrix2D<TElement> d = Machines[0].GPU.Upload(local))
                {
                    encoded = Machines[i].Encode(d);
                }

                layerTrainData = encoded;

                if (layerTrainData.GetLength(0) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }


            layerTrainData.Dispose();
        }


        public void SetDefaultMachineState(SuspendState state)
        {
            foreach (var machine in Machines)
            {
                machine.SetState(state);
            }
        }


        TElement[,] INetwork<TElement>.Daydream(int numDreams, int maxDepth = -1, bool guassian = true)
        {
            throw new NotImplementedException();
        }

        TElement[,] INetwork<TElement>.DaydreamWithLabels(int numDreams, out TElement[,] labels, bool guassian = true,
            bool softmaxLabels = true)
        {
            throw new NotImplementedException();
        }

        TElement[,] INetwork<TElement>.DaydreamByClass(TElement[,] modelLabels, out TElement[,] generatedLabels,
            bool guassian = true, bool softmaxGeneratedLabels = true)
        {
            throw new NotImplementedException();
        }

        protected void OnEpochComplete(EpochEventArgs<TElement> args)
        {
            if (EpochComplete != null)
            {
                EpochComplete(this, args);
            }
        }

        protected void OnLayerTrainComplete(EpochEventArgs<TElement> args)
        {
            if (LayerTrainComplete != null)
            {
                LayerTrainComplete(this, args);
            }
        }

        public TElement[,] Daydream(TElement noiseScale, int numDreams, int maxDepth = -1, bool guassian = true)
        {
            using (Matrix2D<TElement> a = AsCuda.Daydream(noiseScale, numDreams, maxDepth, guassian))
                return a.CopyLocal();
        }

        public TElement[,] DaydreamWithLabels(TElement noiseScale, int numDreams, out TElement[,] labels,
            bool guassian = true, bool softmaxLabels = true)
        {
            Matrix2D<TElement> lbl;
            using (
                Matrix2D<TElement> res = AsCuda.DaydreamWithLabels(noiseScale, numDreams, out lbl, guassian, softmaxLabels)
                )
            using (lbl)
            {
                labels = lbl.CopyLocal();
                return res.CopyLocal();
            }
        }

        public TElement[,] DaydreamByClass(TElement noiseScale, TElement[,] modelLabels,
            out TElement[,] generatedLabels, bool guassian = true, bool softmaxGeneratedLabels = true)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(modelLabels))
            {
                Matrix2D<TElement> gen;
                using (
                    Matrix2D<TElement> res = AsCuda.DaydreamByClass(noiseScale, d, out gen, guassian,
                        softmaxGeneratedLabels))
                using (gen)
                {
                    generatedLabels = gen.CopyLocal();
                    return res.CopyLocal();
                }
            }
        }

        public void GreedyTrain(TElement[,] data,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
                AsCuda.GreedyTrain(d, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory, cancelToken, trainFrom);
        }

        public void GreedySupervisedTrain(TElement[,] data, TElement[,] labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            using (Matrix2D<TElement> l = Machines[0].GPU.Upload(labels))
                AsCuda.GreedySupervisedTrain(d, l, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory, cancelToken, trainFrom);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var machine in Machines)
                {
                    machine.Dispose();
                }
            }
        }

        ~CudaAdvancedNetwork()
        {
            Trace.TraceError("Finalizer called!. Object should be disposed properly");
            Dispose(false);
        }

        public void GreedyBatchedTrain(TElement[,] data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            {
                AsCuda.GreedyBatchedTrain(d, batchSize, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory, cancelToken, trainFrom);
            }
        }

        public void GreedyBatchedSupervisedTrain(TElement[,] data, TElement[,] labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            using (Matrix2D<TElement> l = Machines[0].GPU.Upload(labels))
                AsCuda.GreedyBatchedSupervisedTrainMem(d, l, batchSize, exitConditionFactory,
                    weightLearningRateCalculatorFactory, hidBiasLearningRateCalculatorFactory,
                    visBiasLearningRateCalculatorFactory, cancelToken, trainFrom);
        }

        public void GreedyBatchedTrainMem(TElement[,] data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            {
                AsCuda.GreedyBatchedTrainMem(d, batchSize, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory, cancelToken, trainFrom);
            }
        }

        public void GreedyBatchedSupervisedTrainMem(TElement[,] data, TElement[,] labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            using (Matrix2D<TElement> d = Machines[0].GPU.Upload(data))
            using (Matrix2D<TElement> l = Machines[0].GPU.Upload(labels))
                AsCuda.GreedyBatchedSupervisedTrainMem(d, l, batchSize, exitConditionFactory,
                    weightLearningRateCalculatorFactory, hidBiasLearningRateCalculatorFactory,
                    visBiasLearningRateCalculatorFactory, cancelToken, trainFrom);
        }

        public void GreedyBatchedTrainMem(IList<TElement[,]> batches,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            IList<TElement[,]> layerTrainData = batches;
            int dataLen = batches.Sum(a => a.GetLength(0));

            for (int i = 0; i < Machines.Count; i++)
            {

                cancelToken.ThrowIfCancellationRequested();
                if (i >= trainFrom)
                    Machines[i].GreedyBatchedTrainMem(layerTrainData, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);

                layerTrainData = layerTrainData.Select(Machines[i].Encode).ToList();

                if (layerTrainData.Sum(a => a.GetLength(0)) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
        }

        public void GreedyBatchedSupervisedTrainMem(IList<TElement[,]> batches, IList<TElement[,]> labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken, int trainFrom = 0)
        {
            if (batches.Where((a, i) => a.GetLength(0) != labels[i].GetLength(0)).Any())
            {
                throw new Exception("Mismatch between lengths of batch data and batch labels");
            }
            IList<TElement[,]> layerTrainData = batches;

            int dataLen = batches.Sum(a => a.GetLength(0));

            for (int i = 0; i < Machines.Count; i++)
            {


                cancelToken.ThrowIfCancellationRequested();
                if (i == Machines.Count - 1)
                {
                    try
                    {
                        List<TElement[,]> combined = layerTrainData.Select((a, j) =>
                        {
                            using (Matrix2D<TElement> gp = Machines[0].GPU.Upload(a))
                            using (Matrix2D<TElement> gk = Machines[0].GPU.Upload(labels[j]))
                            using (Matrix2D<TElement> c = Machines[0].GPU.AllocateNoSet<TElement>(gp.GetLength(0),
                                gp.GetLength(1) + gk.GetLength(1)))
                            {
                                c.InsertValuesFrom(0, 0, gp);
                                c.InsertValuesFrom(0, gp.GetLength(1), gk);
                                return c.CopyLocal();
                            }
                        }).ToList();

                        layerTrainData = combined;
                    }
                    catch (AggregateException agg)
                    {
                        if (agg.InnerException is TaskCanceledException ||
                            agg.InnerException is OperationCanceledException)
                            throw agg.InnerException;
                    }
                }

                if (i >= trainFrom)
                    Machines[i].GreedyBatchedTrainMem(layerTrainData, exitConditionFactory.Create(i),
                        weightLearningRateCalculatorFactory.Create(i),
                        hidBiasLearningRateCalculatorFactory.Create(i),
                        visBiasLearningRateCalculatorFactory.Create(i), cancelToken);


                layerTrainData = layerTrainData.Select(Machines[i].Encode).ToList();

                if (layerTrainData.Sum(a => a.GetLength(0)) != dataLen)
                {
                    throw new Exception("Data length has changed");
                }
            }
        }
    }
}