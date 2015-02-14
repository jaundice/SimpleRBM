using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;
using TElement = System.Single;
using LSI = SimpleRBM.Common.Save.LayerSaveInfoF;
using RBM = SimpleRBM.Cuda.CudaRbmF;

namespace SimpleRBM.Cuda
{
    public class CudaDbnF : IBasicNetworkCuda<TElement>
    {
        private readonly IList<IBasicRbmCuda<TElement>> Machines;

        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;


        public CudaDbnF(GPGPU gpu, GPGPURAND rand, IEnumerable<IBasicRbmCuda<TElement>> machines)
        {
            _gpu = gpu;
            _rand = rand;
            Machines = new List<IBasicRbmCuda<TElement>>();
            foreach (var rbm in machines)
            {
                rbm.EpochEnd += (a, b) => RaiseEpochEnd(b);
                rbm.TrainEnd += (a, b) => RaiseTrainEnd(b);
                Machines.Add(rbm);
            }
        }

        public CudaDbnF(GPGPU gpu, GPGPURAND rand, DirectoryInfo network, ILayerDefinition[] appendLayers = null)
        {
            _gpu = gpu;
            _rand = rand;
            List<LSI> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LSI(a.FullName)).ToList();

            appendLayers = appendLayers ?? new ILayerDefinition[0];
            Machines =
                new List<IBasicRbmCuda<TElement>>(saveInfos.Count() + appendLayers.Length);


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new RBM(gpu, rand, saveInfos[i].NumVisible, saveInfos[i].NumHidden,
                    i, saveInfos[i].Weights, saveInfos[i].VisibleActivation, saveInfos[i].HiddenActivation);
                rbm.EpochEnd += (a, b) => RaiseEpochEnd(b);
                rbm.TrainEnd += (a, b) => RaiseTrainEnd(b);
                Machines.Add(rbm);
            }

            if (appendLayers.Length > 0)
            {
                for (int j = 0; j < appendLayers.Length; j++)
                {
                    Console.WriteLine("Appending Layer {0}: {1}x{2}", j + saveInfos.Count,
                        appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits);

                    var rbm = new RBM(gpu, rand,
                        appendLayers[j].VisibleUnits, appendLayers[j + 1].HiddenUnits,
                        saveInfos.Count + j, appendLayers[j].VisibleActivation, saveInfos[j].HiddenActivation);
                    rbm.EpochEnd += (a, b) => RaiseEpochEnd(b);
                    rbm.TrainEnd += (a, b) => RaiseTrainEnd(b);
                    Machines.Add(rbm);
                }
            }

        }

        public CudaDbnF(GPGPU gpu, GPGPURAND rand, ILayerDefinition[] layerSizes)
        {
            _gpu = gpu;
            _rand = rand;

            Machines = new List<IBasicRbmCuda<TElement>>(layerSizes.Length);

            for (int i = 0; i < layerSizes.Length; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits,
                    layerSizes[i].HiddenUnits);

                var rbm = new RBM(gpu, rand, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits,
                    i, layerSizes[i].VisibleActivation, layerSizes[i].HiddenActivation);
                rbm.EpochEnd += (a, b) => RaiseEpochEnd(b);
                rbm.TrainEnd += (a, b) => RaiseTrainEnd(b);
                Machines.Add(rbm);
            }
        }

        public bool Disposed { get; protected set; }

        public int NumMachines
        {
            get { return Machines.Count; }
        }

        public TElement[,] Encode(TElement[,] data)
        {
            using (var d = _gpu.Upload(data))
            using (var ret = ((IBasicNetworkCuda<TElement>)this).Encode(d))
                return ret.CopyLocal();
        }

        public TElement[,] Decode(TElement[,] data)
        {
            return Decode(data, Machines.Count - 1);
        }

        public TElement[,] Decode(TElement[,] data, int maxDepth)
        {
            using (var d = _gpu.Upload(data))
            using (var ret = ((IBasicNetworkCuda<TElement>)this).Decode(d, maxDepth))
                return ret.CopyLocal();
        }

        public TElement[,] Reconstruct(TElement[,] data)
        {
            return Reconstruct(data, Machines.Count - 1);
        }

        public TElement[,] Reconstruct(TElement[,] data, int maxDepth)
        {
            using (var d = _gpu.Upload(data))
            using (var res = ((IBasicNetworkCuda<TElement>)this).Reconstruct(d, maxDepth))
                return res.CopyLocal();
        }

        public TElement[,] ReconstructWithLabels(TElement[,] data, out TElement[,] labels)
        {
            Matrix2D<TElement> lbl;
            using (var d = _gpu.Upload(data))
            using (var ret = ((IBasicNetworkCuda<TElement>)this).ReconstructWithLabels(d, out lbl))
            using (lbl)
            {
                labels = lbl.CopyLocal();
                return ret.CopyLocal();
            }
        }

        public TElement[,] DayDream(int numberOfDreams)
        {
            return DayDream(numberOfDreams, Machines.Count - 1);
        }

        public TElement[,] DayDream(int numberOfDreams, int maxDepth)
        {
            using (var res = ((IBasicNetworkCuda<TElement>)this).DayDream(numberOfDreams, maxDepth))
                return res.CopyLocal();
        }

        public event EventHandler<EpochEventArgs<TElement>> EpochEnd;

        public event EventHandler<EpochEventArgs<TElement>> TrainEnd;

        public IEnumerable<ILayerSaveInfo<TElement>> GetLayerSaveInfos()
        {
            return Machines.Select(restrictedBoltzmannMachineF => restrictedBoltzmannMachineF.GetSaveInfo());
        }


        public void GreedyTrainAll(TElement[,] visibleData,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            using (var d = _gpu.Upload(visibleData))
            {
                ((IBasicNetworkCuda<TElement>)this).GreedyTrainAll(d, exitConditionEvaluatorFactory,
                    learningRateFactory);
            }
        }


        void IBasicNetworkCuda<TElement>.UpDownTrainAll(Matrix2D<TElement> visibleData, int iterations,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;

            for (int i = 0; i < iterations; i++)
            {
                Matrix2D<TElement> penultimateActivations = ((IBasicNetworkCuda<TElement>)this).Encode(visibleData, Machines.Count - 2);

                Machines[Machines.Count - 1].GreedyTrain(penultimateActivations, exitConditionEvaluatorFactory.Create(Machines.Count - 1),
                    learningRateFactory.Create(Machines.Count - 1));



                var visible = Machines[Machines.Count - 1].Reconstruct(penultimateActivations);
                penultimateActivations.Dispose();

                for (int j = Machines.Count - 2; j > -1; j--)
                {
                    Machines[j].DownPass(visible, exitConditionEvaluatorFactory.Create(j), learningRateFactory.Create(j),
                        out error);

                    var visible2 = Machines[j].Decode(visible);
                    visible.Dispose();
                    visible = visible2;
                }
                visible.Dispose();
            }
        }

        void IBasicNetworkCuda<TElement>.UpDownSupervisedTrainAll(Matrix2D<TElement> visibleData, Matrix2D<TElement> labels, int iterations,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;

            for (int i = 0; i < iterations; i++)
            {
                Matrix2D<TElement> penultimateActivations = ((IBasicNetworkCuda<TElement>)this).Encode(visibleData, Machines.Count - 2);

                var combined = Machines[0].GPU.AllocateNoSet<TElement>(penultimateActivations.GetLength(0),
                    penultimateActivations.GetLength(1) + labels.GetLength(1));

                combined.InsertValuesFrom(0, 0, penultimateActivations);
                combined.InsertValuesFrom(0, penultimateActivations.GetLength(1), labels);
                penultimateActivations.Dispose();
                penultimateActivations = combined;


                Machines[Machines.Count - 1].GreedyTrain(penultimateActivations, exitConditionEvaluatorFactory.Create(Machines.Count - 1),
                    learningRateFactory.Create(Machines.Count - 1));



                var visible = Machines[Machines.Count - 1].Reconstruct(penultimateActivations);
                penultimateActivations.Dispose();
                //labels = visible.SubMatrix(0, Machines[Machines.Count - 2].NumHiddenNeurons);
                var c = visible.SubMatrix(0, 0, 0, Machines[Machines.Count - 2].NumHiddenNeurons);
                visible.Dispose();
                visible = c;

                for (int j = Machines.Count - 2; j > -1; j--)
                {
                    Machines[j].DownPass(visible, exitConditionEvaluatorFactory.Create(j), learningRateFactory.Create(j),
                        out error);

                    var visible2 = Machines[j].Decode(visible);
                    visible.Dispose();
                    visible = visible2;
                }
                visible.Dispose();
            }
        }


        public TElement[,] DaydreamByClass(TElement[,] labels)
        {
            Matrix2D<TElement> lbl;
            using (var d = _gpu.Upload(labels))

            using (var ret = ((IBasicNetworkCuda<TElement>)this).DaydreamByClass(d, out lbl, true, false))
            {
                return ret.CopyLocal();
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


        private void RaiseTrainEnd(EpochEventArgs<TElement> args)
        {
            if (TrainEnd != null)
                TrainEnd(this, args);
        }


        private void RaiseEpochEnd(EpochEventArgs<TElement> e)
        {
            if (EpochEnd != null)
                EpochEnd(this, e);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (IBasicRbmCuda<TElement> restrictedBoltzmannMachineF in Machines)
                {
                    restrictedBoltzmannMachineF.Dispose();
                }
                _rand.Dispose();
                _gpu.FreeAll();
                _gpu.UnloadModules();
                _gpu.Dispose();
            }
        }

        ~CudaDbnF()
        {
            Dispose(false);
        }

        IList<IBasicRbmCuda<TElement>> IBasicNetworkCuda<TElement>.Machines
        {
            get { return Machines; }
        }

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.Encode(Matrix2D<TElement> data, int maxDepth = -1)
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

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.Decode(Matrix2D<TElement> activations, int maxDepth = -1)
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

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.Reconstruct(Matrix2D<TElement> data, int maxDepth = -1)
        {
            var depth = maxDepth == -1 ? Machines.Count - 1 : maxDepth;

            using (var encoded = ((IBasicNetworkCuda<TElement>)this).Encode(data, depth))
                return ((IBasicNetworkCuda<TElement>)this).Decode(encoded, depth);
        }

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.DayDream(int numberOfDreams, int maxDepth = -1,
            bool guassian = true)
        {
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, numberOfDreams,
                        Machines[0].NumVisibleNeurons, (TElement)0.5, (TElement)0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, numberOfDreams,
                        Machines[0].NumVisibleNeurons, (TElement)1))
            {
                return ((IBasicNetworkCuda<TElement>)this).Reconstruct(rand, maxDepth);
            }
        }


        void IBasicNetworkCuda<TElement>.GreedyTrainAll(Matrix2D<TElement> visibleData,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            ((IBasicNetworkCuda<TElement>)this).GreedyTrainLayersFrom(visibleData, -1, exitConditionEvaluatorFactory,
                learningRateFactory);
        }

        void IBasicNetworkCuda<TElement>.GreedyTrainLayersFrom(Matrix2D<TElement> visibleData, int startDepth,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            var layerTrainData = visibleData;
            for (var i = 0; i < Machines.Count; i++)
            {
                if (i >= startDepth)
                {
                    Machines[i].GreedyTrain(layerTrainData, exitConditionEvaluatorFactory.Create(i),
                        learningRateFactory.Create(i));
                }
                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, visibleData))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }
        }

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.DaydreamByClass(Matrix2D<TElement> modelLabels,
            out Matrix2D<TElement> generatedLabels, bool guassian, bool softmaxLabels)
        {
            var highest = Machines.Count - 1;
            using (
                var rand = guassian
                    ? Machines[0].GPU.GuassianDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, (TElement)0.5, (TElement)0.2)
                    : Machines[0].GPU.UniformDistribution(Machines[0].GPURAND, modelLabels.GetLength(0),
                        Machines[highest].NumVisibleNeurons, (TElement)1))
            {
                rand.InsertValuesFrom(0, Machines[highest - 1].NumHiddenNeurons, modelLabels);
                using (var encoded = Machines[highest].Encode(rand))
                {
                    return ((IBasicNetworkCuda<TElement>)this).DecodeWithLabels(encoded, out generatedLabels,
                        softmaxLabels);
                }
            }
        }

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.DecodeWithLabels(Matrix2D<TElement> activations,
            out Matrix2D<TElement> labels, bool softmaxLabels)
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

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.EncodeWithLabelExpansion(Matrix2D<TElement> data)
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

        Matrix2D<TElement> IBasicNetworkCuda<TElement>.ReconstructWithLabels(Matrix2D<TElement> data,
            out Matrix2D<TElement> labels, bool softmaxLabels)
        {
            var depth = Machines.Count - 1;

            using (var d = ((IBasicNetworkCuda<TElement>)this).EncodeWithLabelExpansion(data))
                return ((IBasicNetworkCuda<TElement>)this).DecodeWithLabels(d, out labels, softmaxLabels);
        }

        void IBasicNetworkCuda<TElement>.GreedyBatchedTrainAll(Matrix2D<TElement> data,
            int batchRows, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory, out TElement error)
        {
            error = 0;
            var layerTrainData = data;
            for (var i = 0; i < Machines.Count; i++)
            {

                error = Machines[i].GreedyBatchedTrain(layerTrainData, batchRows, exitConditionEvaluatorFactory.Create(i),
                     learningRateFactory.Create(i));

                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }
            layerTrainData.Dispose();
        }

        void IBasicNetworkCuda<TElement>.GreedySupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels, IExitConditionEvaluatorFactory<TElement> exitConditionFactory, ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory)
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
                    weightLearningRateCalculatorFactory.Create(i));


                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }
            layerTrainData.Dispose();

        }

        void IBasicNetworkCuda<TElement>.GreedyBatchedSupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels, int batchRows, IExitConditionEvaluatorFactory<TElement> exitConditionFactory, ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory)
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


                Machines[i].GreedyBatchedTrain(layerTrainData,batchRows, exitConditionFactory.Create(i),
                    weightLearningRateCalculatorFactory.Create(i));


                var encoded = Machines[i].Encode(layerTrainData);
                if (!ReferenceEquals(layerTrainData, data))
                {
                    layerTrainData.Dispose();
                }
                layerTrainData = encoded;
            }
            layerTrainData.Dispose();

        }

        public TElement[,] Encode(TElement[,] data, int maxDepth)
        {
            using (var d = _gpu.Upload(data))
            using (var ret = ((IBasicNetworkCuda<TElement>)this).Encode(d, maxDepth))
                return ret.CopyLocal();
        }

        public void GreedySupervisedTrainAll(TElement[,] srcData, TElement[,] labels, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            using (var d = _gpu.Upload(srcData))
            using (var l = _gpu.Upload(labels))
            {
                ((IBasicNetworkCuda<TElement>)this).GreedySupervisedTrain(d, l, exitConditionEvaluatorFactory,
                    learningRateFactory);
            }
        }


        public void UpDownTrainAll(TElement[,] visibleData, int iterations, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            using (var d = _gpu.Upload(visibleData))
            {
                ((IBasicNetworkCuda<TElement>)this).UpDownTrainAll(d, iterations, exitConditionEvaluatorFactory, learningRateFactory);
            }
        }

        public void UpDownTrainSupervisedAll(TElement[,] visibleData, TElement[,] labels, int iterations, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            using (var d = _gpu.Upload(visibleData))
            using (var l = _gpu.Upload(labels))
            {
                ((IBasicNetworkCuda<TElement>)this).UpDownSupervisedTrainAll(d, l, iterations, exitConditionEvaluatorFactory, learningRateFactory);
            }
        }


        public void GreedyBatchedSupervisedTrainAll(TElement[,] visibleData, TElement[,] labels, int batchSize, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            using (var d = _gpu.Upload(visibleData))
            using (var l = _gpu.Upload(labels))
            {
                ((IBasicNetworkCuda<TElement>)this).GreedyBatchedSupervisedTrain(d, l, batchSize, exitConditionEvaluatorFactory, learningRateFactory);
            }
        }


        public void GreedyBatchedTrainAll(TElement[,] visibleData, int batchRows, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            using (var d = _gpu.Upload(visibleData))
            {
                TElement error;
                ((IBasicNetworkCuda<TElement>)this).GreedyBatchedTrainAll(d, batchRows, exitConditionEvaluatorFactory, learningRateFactory, out error);
            }
        }
    }
}