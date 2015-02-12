using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ICSharpCode.Decompiler.Ast.Transforms;
using SimpleRBM.Common;
using SimpleRBM.Cuda;
#if USEFLOAT
using TElementType = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElementType = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;
#endif

namespace CudaNN
{
    public class Network : IDisposable
    {
        public IList<RbmBase> Machines { get; protected set; }
        public event EventHandler<EpochEventArgs<TElementType>> EpochComplete;
        public event EventHandler<EpochEventArgs<TElementType>> LayerTrainComplete;

        protected void OnEpochComplete(EpochEventArgs<TElementType> args)
        {
            if (EpochComplete != null)
            {
                EpochComplete(this, args);
            }
        }

        protected void OnLayerTrainComplete(EpochEventArgs<TElementType> args)
        {
            if (LayerTrainComplete != null)
            {
                LayerTrainComplete(this, args);
            }
        }

        public Network(IEnumerable<RbmBase> machines)
        {
            Machines = machines.ToList();

            foreach (var machine in Machines)
            {
                machine.EpochComplete += (a, b) => OnEpochComplete(b);
                machine.TrainComplete += (a, b) => OnLayerTrainComplete(b);
            }
        }


        public TElementType[,] Reconstruct(TElementType[,] data, int maxDepth = -1)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var res = Reconstruct(d, maxDepth))
                return res.CopyLocal();
        }

        public TElementType[,] Encode(TElementType[,] data, int maxDepth = -1)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var res = Encode(d, maxDepth))
                return res.CopyLocal();
        }

        public Matrix2D<TElementType> Encode(Matrix2D<TElementType> data, int maxDepth = -1)
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
        private Matrix2D<TElementType> EncodeWithLabelExpansion(Matrix2D<TElementType> data)
        {
            var d = data;
            var depth = Machines.Count - 1;
            for (var i = 0; i < depth + 1; i++)
            {
                if (i == Machines.Count - 1)
                {
                    var expanded = Machines[0].GPU.AllocateAndSet<TElementType>(data.GetLength(0),
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

        public Matrix2D<TElementType> Reconstruct(Matrix2D<TElementType> data, int maxDepth = -1)
        {
            var depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;

            using (var encoded = Encode(data, depth))
                return Decode(encoded, depth);
        }


        public TElementType[,] Decode(TElementType[,] activations, int maxDepth = -1)
        {
            using (var act = Machines[0].GPU.Upload(activations))
            {
                using (var res = Decode(act, maxDepth))
                {
                    return res.CopyLocal();
                }
            }
        }


        public Matrix2D<TElementType> Decode(Matrix2D<TElementType> activations, int maxDepth = -1)
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

        public TElementType[,] ReconstructWithLabels(TElementType[,] data,
            out TElementType[,] labels)
        {
            Matrix2D<TElementType> lbl;
            using (var d = Machines[0].GPU.Upload(data))
            using (var recon = ReconstructWithLabels(d, out lbl))
            using (lbl)
            {
                labels = lbl.CopyLocal();
                return recon.CopyLocal();
            }
        }


        public Matrix2D<TElementType> ReconstructWithLabels(Matrix2D<TElementType> data, out Matrix2D<TElementType> labels)
        {
            var depth = Machines.Count - 1;

            using (var d = EncodeWithLabelExpansion(data))
                return DecodeWithLabels(d, out labels);
        }

        public TElementType[,] DecodeWithLabels(TElementType[,] activations,
            out TElementType[,] labels)
        {
            using (var act = Machines[0].GPU.Upload(activations))
            {
                Matrix2D<TElementType> lbl;
                using (var res = DecodeWithLabels(act, out lbl))
                using (lbl)
                {
                    labels = lbl.CopyLocal();
                    return res.CopyLocal();
                }
            }
        }


        public Matrix2D<TElementType> DecodeWithLabels(Matrix2D<TElementType> activations,
            out Matrix2D<TElementType> labels)
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
                    var c = constructed.SubMatrix(0, 0, 0, Machines[i - 1].NumHiddenNeurons);
                    constructed.Dispose();
                    constructed = c;
                }
                if (!ReferenceEquals(d, activations))
                    d.Dispose();
                d = constructed;
            }
            return d;
        }


        public TElementType[,] LabelData(TElementType[,] data)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var r = LabelData(d))
                return r.CopyLocal();
        }

        public Matrix2D<TElementType> LabelData(Matrix2D<TElementType> data)
        {
            var depth = Machines.Count - 1;
            using (var d = EncodeWithLabelExpansion(data))
            using (var constructed = Machines[depth].Decode(d))
            {
                return constructed.SubMatrix(0, Machines[depth - 1].NumHiddenNeurons);
            }
        }

        public TElementType[,] DaydreamM(int numDreams, int maxDepth = -1, bool guassian = true)
        {
            using (var a = Daydream(numDreams, maxDepth, guassian))
                return a.CopyLocal();
        }

        public Matrix2D<TElementType> Daydream(int numDreams, int maxDepth = -1, bool guassian = true)
        {
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution( Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElementType)0.5, (TElementType)0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElementType)1))
            {
                return Reconstruct(rand, maxDepth);
            }
        }

        public TElementType[,] DaydreamWithLabels(int numDreams, out TElementType[,] labels,
            bool guassian = true)
        {
            Matrix2D<TElementType> lbl;
            using (var res = DaydreamWithLabels(numDreams, out lbl, guassian))
            using (lbl)
            {
                labels = lbl.CopyLocal();
                return res.CopyLocal();
            }
        }

        public Matrix2D<TElementType> DaydreamWithLabels(int numDreams, out Matrix2D<TElementType> labels, bool guassian = true)
        {
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElementType)0.5, (TElementType)0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, numDreams,
                        Machines[0].NumVisibleNeurons, (TElementType)1))
            {
                return ReconstructWithLabels(rand, out labels);
            }
        }

        public TElementType[,] DaydreamByClass(TElementType[,] modelLabels,
            out TElementType[,] generatedLabels, bool guassian = true)
        {
            using (var d = Machines[0].GPU.Upload(modelLabels))
            {
                Matrix2D<TElementType> gen;
                using (var res = DaydreamByClass(d, out gen, false))
                using (gen)
                {
                    generatedLabels = gen.CopyLocal();
                    return res.CopyLocal();
                }
            }
        }

        public Matrix2D<TElementType> DaydreamByClass(Matrix2D<TElementType> modelLabels, out Matrix2D<TElementType> generatedLabels, bool guassian = true)
        {
            var highest = Machines.Count - 1;
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, (TElementType)0.5, (TElementType)0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, (TElementType)1))
            //using (var rand = Machines[0].GPU.AllocateAndSet<TElementType>(modelLabels.GetLength(0),
            //            Machines[highest].NumVisibleNeurons))
            {
                rand.InsertValuesFrom(0, Machines[highest - 1].NumHiddenNeurons, modelLabels);
                using (var encoded = Machines[highest].Encode(rand))
                {
                    return DecodeWithLabels(encoded, out generatedLabels);
                }
            }
        }

        public void GreedyTrain(TElementType[,] data,
            IExitConditionEvaluatorFactory<TElementType> exitConditionFactory,
            ILearningRateCalculatorFactory<TElementType> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> visBiasLearningRateCalculatorFactory)
        {
            using (var d = Machines[0].GPU.Upload(data))
                GreedyTrain(d, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory);
        }

        public void GreedySupervisedTrain(TElementType[,] data, TElementType[,] labels,
           IExitConditionEvaluatorFactory<TElementType> exitConditionFactory,
           ILearningRateCalculatorFactory<TElementType> weightLearningRateCalculatorFactory,
           ILearningRateCalculatorFactory<TElementType> hidBiasLearningRateCalculatorFactory,
           ILearningRateCalculatorFactory<TElementType> visBiasLearningRateCalculatorFactory)
        {
            using (var d = Machines[0].GPU.Upload(data))
            using (var l = Machines[0].GPU.Upload(labels))
                GreedySupervisedTrain(d, l, exitConditionFactory, weightLearningRateCalculatorFactory,
                    hidBiasLearningRateCalculatorFactory, visBiasLearningRateCalculatorFactory);

        }


        public void GreedyTrain(Matrix2D<TElementType> data,
            IExitConditionEvaluatorFactory<TElementType> exitConditionFactory,
            ILearningRateCalculatorFactory<TElementType> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {
                Machines[i].GreedyTrain(layerTrainData, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(Machines[i].LayerIndex),
                    hidBiasLearningRateCalculatorFactory.Create(Machines[i].LayerIndex),
                    visBiasLearningRateCalculatorFactory.Create(Machines[i].LayerIndex));
                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }
        }

        public void GreedySupervisedTrain(Matrix2D<TElementType> data, Matrix2D<TElementType> labels,
            IExitConditionEvaluatorFactory<TElementType> exitConditionFactory,
            ILearningRateCalculatorFactory<TElementType> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> visBiasLearningRateCalculatorFactory)
        {
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {

                if (i == Machines.Count - 1)
                {
                    var combined = Machines[0].GPU.AllocateNoSet<TElementType>(data.GetLength(0),
                        layerTrainData.GetLength(1) + labels.GetLength(1));

                    combined.InsertValuesFrom(0, 0, layerTrainData);
                    combined.InsertValuesFrom(0, layerTrainData.GetLength(1), labels);
                    layerTrainData.Dispose();
                    layerTrainData = combined;
                }


                Machines[i].GreedyTrain(layerTrainData, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(Machines[i].LayerIndex),
                    hidBiasLearningRateCalculatorFactory.Create(Machines[i].LayerIndex),
                    visBiasLearningRateCalculatorFactory.Create(Machines[i].LayerIndex));


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

        public void Dispose()
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

        public bool Disposed { get; protected set; }

        ~Network()
        {
            Trace.TraceError("Finalizer called!. Object should be disposed properly");
            Dispose(false);
        }
    }
}