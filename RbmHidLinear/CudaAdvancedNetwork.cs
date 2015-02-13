using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
        public IList<IAdvancedRbmCuda<TElement>> Machines { get; protected set; }
        public event EventHandler<EpochEventArgs<TElement>> EpochComplete;
        public event EventHandler<EpochEventArgs<TElement>> LayerTrainComplete;

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

        public CudaAdvancedNetwork(IEnumerable<IAdvancedRbmCuda<TElement>> machines)
        {
            Machines = machines.ToList();

            foreach (var machine in Machines)
            {
                machine.EpochEnd += (a, b) => OnEpochComplete(b);
                machine.TrainEnd += (a, b) => OnLayerTrainComplete(b);
            }
        }


        public TElement[,] Reconstruct(TElement[,] data, int maxDepth = -1)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var res = AsCuda.Reconstruct(d, maxDepth))
                return res.CopyLocal();
        }

        public TElement[,] Encode(TElement[,] data, int maxDepth = -1)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var res = AsCuda.Encode(d, maxDepth))
                return res.CopyLocal();
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.Encode(Matrix2D<TElement> data, int maxDepth = -1)
        {
            var d = data;
            var depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;
            for (var i = 0; i < depth + 1; i++)
            {
                var encoded = Machines[i].Encode(d);
                if (!ReferenceEquals(d, data))
                {
                    d.Dispose();
                }
                d = encoded;
            }
            return d;
        }

        /// <summary>
        /// Same as Encode except the visible data is extended for the inner most rbm to the size of the visible buffer (i.e numlabels columns are added)
        /// </summary>
        /// <param name="data"></param>
        /// <param name="maxDepth"></param>
        /// <returns></returns>
        Matrix2D<TElement> ICudaNetwork<TElement>.EncodeWithLabelExpansion(Matrix2D<TElement> data)
        {
            var d = data;
            var depth = Machines.Count - 1;
            for (var i = 0; i < depth + 1; i++)
            {
                if (i == Machines.Count - 1)
                {
                    var expanded = Machines[0].GPU.AllocateAndSet<TElement>(data.GetLength(0),
                        Machines[i].NumVisibleNeurons);
                    expanded.InsertValuesFrom(0, 0, d);

                    if (!ReferenceEquals(d, data))
                        d.Dispose();

                    d = expanded;
                }

                var encoded = Machines[i].Encode(d);
                if (!ReferenceEquals(d, data))
                {
                    d.Dispose();
                }
                d = encoded;
            }
            return d;
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.Reconstruct(Matrix2D<TElement> data, int maxDepth = -1)
        {
            var depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;

            using (var encoded = AsCuda.Encode(data, depth))
                return AsCuda.Decode(encoded, depth);
        }


        public TElement[,] Decode(TElement[,] activations, int maxDepth = -1)
        {
            using (var act = Machines[0].GPU.Upload(activations))
            {
                using (var res = AsCuda.Decode(act, maxDepth))
                {
                    return res.CopyLocal();
                }
            }
        }


        Matrix2D<TElement> ICudaNetwork<TElement>.Decode(Matrix2D<TElement> activations, int maxDepth = -1)
        {
            var depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;

            var d = activations;

            for (var i = depth; i > -1; i--)
            {
                var constructed = Machines[i].Decode(d);
                if (!ReferenceEquals(d, activations))
                    d.Dispose();
                d = constructed;
            }
            return d;
        }

        public TElement[,] ReconstructWithLabels(TElement[,] data,
            out TElement[,] labels, bool softmaxLabels = true)
        {
            Matrix2D<TElement> lbl;
            using (var d = Machines[0].GPU.Upload(data))
            using (var recon = AsCuda.ReconstructWithLabels(d, out lbl, softmaxLabels))
            using (lbl)
            {
                labels = lbl.CopyLocal();
                return recon.CopyLocal();
            }
        }


        Matrix2D<TElement> ICudaNetwork<TElement>.ReconstructWithLabels(Matrix2D<TElement> data,
            out Matrix2D<TElement> labels, bool softmaxLabels = true)
        {
            var depth = Machines.Count - 1;

            using (var d = AsCuda.EncodeWithLabelExpansion(data))
                return AsCuda.DecodeWithLabels(d, out labels, softmaxLabels);
        }

        public TElement[,] DecodeWithLabels(TElement[,] activations,
            out TElement[,] labels, bool softmaxLabels = true)
        {
            using (var act = Machines[0].GPU.Upload(activations))
            {
                Matrix2D<TElement> lbl;
                using (var res = AsCuda.DecodeWithLabels(act, out lbl, softmaxLabels))
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
            var depth = Machines.Count - 1;

            labels = null;
            var d = activations;
            for (var i = depth; i > -1; i--)
            {
                var constructed = Machines[i].Decode(d);

                if (i == depth)
                {
                    labels = constructed.SubMatrix(0, Machines[i - 1].NumHiddenNeurons);
                    if (softmaxLabels)
                    {
                        using (labels)
                        {
                            var sm = labels.SoftMax();
                            //sm.ToBinary();
                            labels = sm;
                        }
                    }
                    var c = constructed.SubMatrix(0, 0, 0, Machines[i - 2].NumHiddenNeurons);
                    constructed.Dispose();
                    constructed = c;
                }
                if (!ReferenceEquals(d, activations))
                    d.Dispose();
                d = constructed;
            }
            return d;
        }


        public TElement[,] LabelData(TElement[,] data, bool softmaxLabels = true)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var r = AsCuda.LabelData(d, softmaxLabels))
                return r.CopyLocal();
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.LabelData(Matrix2D<TElement> data, bool softmaxLabels = true)
        {
            var depth = Machines.Count - 1;
            using (var d = AsCuda.EncodeWithLabelExpansion(data))
            using (var constructed = Machines[depth].Decode(d))
            {
                var ret = constructed.SubMatrix(0, Machines[depth - 1].NumHiddenNeurons);
                if (softmaxLabels)
                {
                    using (ret)
                    {
                        var sm = ret.SoftMax();
                        //sm.ToBinary();
                        return sm;
                    }
                }
                else
                {
                    return ret;
                }
            }
        }

        public TElement[,] Daydream(int numDreams, int maxDepth = -1, bool guassian = true)
        {
            using (var a = AsCuda.Daydream(numDreams, maxDepth, guassian))
                return a.CopyLocal();
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.Daydream(int numDreams, int maxDepth = -1, bool guassian = true)
        {
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElement) 0.5, (TElement) 0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElement) 1))
            {
                return AsCuda.Reconstruct(rand, maxDepth);
            }
        }

        public TElement[,] DaydreamWithLabels(int numDreams, out TElement[,] labels,
            bool guassian = true, bool softmaxLabels = true)
        {
            Matrix2D<TElement> lbl;
            using (var res = AsCuda.DaydreamWithLabels(numDreams, out lbl, guassian, softmaxLabels))
            using (lbl)
            {
                labels = lbl.CopyLocal();
                return res.CopyLocal();
            }
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.DaydreamWithLabels(int numDreams,
            out Matrix2D<TElement> labels, bool guassian = true, bool softmaxLabels = true)
        {
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElement) 0.5, (TElement) 0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElement) 1))
            {
                return AsCuda.ReconstructWithLabels(rand, out labels, softmaxLabels);
            }
        }

        public TElement[,] DaydreamByClass(TElement[,] modelLabels,
            out TElement[,] generatedLabels, bool guassian = true, bool softmaxGeneratedLabels = true)
        {
            using (var d = Machines[0].GPU.Upload(modelLabels))
            {
                Matrix2D<TElement> gen;
                using (var res = AsCuda.DaydreamByClass(d, out gen, guassian, softmaxGeneratedLabels))
                using (gen)
                {
                    generatedLabels = gen.CopyLocal();
                    return res.CopyLocal();
                }
            }
        }

        Matrix2D<TElement> ICudaNetwork<TElement>.DaydreamByClass(Matrix2D<TElement> modelLabels,
            out Matrix2D<TElement> generatedLabels, bool guassian = true, bool softmaxLabels = true)
        {
            var highest = Machines.Count - 1;
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, (TElement) 0.5, (TElement) 0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, (TElement) 1))
            {
                rand.InsertValuesFrom(0, Machines[highest - 1].NumHiddenNeurons, modelLabels);
                using (var encoded = Machines[highest].Encode(rand))
                {
                    return AsCuda.DecodeWithLabels(encoded, out generatedLabels, softmaxLabels);
                }
            }
        }

        public void GreedyTrain(TElement[,] data,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            using (var d = Machines[0].GPU.Upload(data))
                AsCuda.GreedyTrain(d, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory);
        }

        public void GreedySupervisedTrain(TElement[,] data, TElement[,] labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var l = Machines[0].GPU.Upload(labels))
                AsCuda.GreedySupervisedTrain(d, l, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory);
        }


        void ICudaNetwork<TElement>.GreedyTrain(Matrix2D<TElement> data,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {
                Machines[i].GreedyTrain(layerTrainData, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(i),
                    hidBiasLearningRateCalculatorFactory.Create(i),
                    visBiasLearningRateCalculatorFactory.Create(i));
                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }
        }

        void ICudaNetwork<TElement>.GreedySupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {
                if (i == Machines.Count - 1)
                {
                    var combined = Machines[0].GPU.AllocateNoSet<TElement>(data.GetLength(0),
                        layerTrainData.GetLength(1) + labels.GetLength(1));

                    combined.InsertValuesFrom(0, 0, layerTrainData);
                    combined.InsertValuesFrom(0, layerTrainData.GetLength(1), labels);
                    layerTrainData.Dispose();
                    layerTrainData = combined;
                }


                Machines[i].GreedyTrain(layerTrainData, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(i),
                    hidBiasLearningRateCalculatorFactory.Create(i),
                    visBiasLearningRateCalculatorFactory.Create(i));


                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
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

        private bool Disposed { get; set; }

        protected ICudaNetwork<TElement> AsCuda
        {
            get { return this; }
        }

        IList<IRestrictedBoltzmannMachine<TElement>> INetwork<TElement>.Machines
        {
            get { return (IList<IRestrictedBoltzmannMachine<TElement>>) Machines; }
        }

        ~CudaAdvancedNetwork()
        {
            Trace.TraceError("Finalizer called!. Object should be disposed properly");
            Dispose(false);
        }


        void ICudaNetwork<TElement>.GreedyBatchedTrain(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {
                Machines[i].GreedyBatchedTrain(layerTrainData, batchSize, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(i),
                    hidBiasLearningRateCalculatorFactory.Create(i),
                    visBiasLearningRateCalculatorFactory.Create(i));
                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }
        }

        void ICudaNetwork<TElement>.GreedyBatchedSupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            int batchSize, IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {
                if (i == Machines.Count - 1)
                {
                    var combined = Machines[0].GPU.AllocateNoSet<TElement>(data.GetLength(0),
                        layerTrainData.GetLength(1) + labels.GetLength(1));

                    combined.InsertValuesFrom(0, 0, layerTrainData);
                    combined.InsertValuesFrom(0, layerTrainData.GetLength(1), labels);
                    layerTrainData.Dispose();
                    layerTrainData = combined;
                }


                Machines[i].GreedyBatchedTrain(layerTrainData, batchSize, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(i),
                    hidBiasLearningRateCalculatorFactory.Create(i),
                    visBiasLearningRateCalculatorFactory.Create(i));


                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }

            if (!ReferenceEquals(layerTrainData, data))
            {
                layerTrainData.Dispose();
            }
        }


        public void GreedyBatchedTrain(TElement[,] data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            //todo: consider managing batch partitions here in main memory allowing for bigger datasets but at the expense of more copying to and from the gpu
            using (var d = Machines[0].GPU.Upload(data))
            {
                AsCuda.GreedyBatchedTrain(d, batchSize, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory);
            }
        }

        public void GreedyBatchedSupervisedTrain(TElement[,] data, TElement[,] labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            //todo: consider managing batch partitions here in main memory allowing for bigger datasets but at the expense of more copying to and from the gpu
            using (var d = Machines[0].GPU.Upload(data))
            using (var l = Machines[0].GPU.Upload(labels))
                AsCuda.GreedyBatchedSupervisedTrain(d, l, batchSize, exitConditionFactory,
                    weightLearningRateCalculatorFactory, hidBiasLearningRateCalculatorFactory,
                    visBiasLearningRateCalculatorFactory);
        }


        void ICudaNetwork<TElement>.GreedyBatchedTrainMem(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {
                var localCopy = layerTrainData.CopyLocal();

                Machines[i].GreedyBatchedTrainMem(layerTrainData, batchSize, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(i),
                    hidBiasLearningRateCalculatorFactory.Create(i),
                    visBiasLearningRateCalculatorFactory.Create(i));

                Matrix2D<float> encoded;
                using (var up = Machines[0].GPU.Upload(localCopy))
                {
                    encoded = Machines[i].Encode(up);
                }

                layerTrainData = encoded;
            }
        }

        void ICudaNetwork<TElement>.GreedyBatchedSupervisedTrainMem(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            int batchSize, IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            TElement[,] local = data.CopyLocal();
            for (var i = 0; i < Machines.Count; i++)
            {
                if (i == Machines.Count - 1)
                {
                    var combined = Machines[0].GPU.AllocateNoSet<TElement>(data.GetLength(0),
                        layerTrainData.GetLength(1) + labels.GetLength(1));

                    combined.InsertValuesFrom(0, 0, layerTrainData);
                    combined.InsertValuesFrom(0, layerTrainData.GetLength(1), labels);
                    labels.Dispose();
                    layerTrainData.Dispose();
                    layerTrainData = combined;
                    local = combined.CopyLocal();
                }


                Machines[i].GreedyBatchedTrainMem(layerTrainData, batchSize, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(i),
                    hidBiasLearningRateCalculatorFactory.Create(i),
                    visBiasLearningRateCalculatorFactory.Create(i));

                Matrix2D<float> encoded;
                using (var d = Machines[0].GPU.Upload(local))
                {
                    encoded = Machines[i].Encode(d);
                }

                layerTrainData = encoded;
            }


            layerTrainData.Dispose();
        }


        public void GreedyBatchedTrainMem(TElement[,] data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            using (var d = Machines[0].GPU.Upload(data))
            {
                AsCuda.GreedyBatchedTrainMem(d, batchSize, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory);
            }
        }

        public void GreedyBatchedSupervisedTrainMem(TElement[,] data, TElement[,] labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory)
        {
            //todo: consider managing batch partitions here in main memory allowing for bigger datasets but at the expense of more copying to and from the gpu
            using (var d = Machines[0].GPU.Upload(data))
            using (var l = Machines[0].GPU.Upload(labels))
                AsCuda.GreedyBatchedSupervisedTrainMem(d, l, batchSize, exitConditionFactory,
                    weightLearningRateCalculatorFactory, hidBiasLearningRateCalculatorFactory,
                    visBiasLearningRateCalculatorFactory);
        }

        public void SetDefaultMachineState(SuspendState state)
        {
            foreach (var machine in Machines)
            {
                machine.SetState(state);
            }
        }
    }
}