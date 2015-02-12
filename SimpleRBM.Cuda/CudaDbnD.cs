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
using TElement = System.Double;
using LSI = SimpleRBM.Common.Save.LayerSaveInfoD;
using RBM = SimpleRBM.Cuda.CudaRbmD;

namespace SimpleRBM.Cuda
{
    public class CudaDbnD : IDeepBeliefNetworkExtended<TElement>, IDisposable
    {
        private readonly RBM[] Machines;

        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;


        public CudaDbnD(GPGPU gpu, GPGPURAND rand, DirectoryInfo network, ILayerDefinition[] appendLayers = null)
        {
            _gpu = gpu;
            _rand = rand;
            //ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            List<LSI> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LSI(a.FullName)).ToList();

            appendLayers = appendLayers ?? new ILayerDefinition[0];
            Machines =
                new RBM[saveInfos.Count() + appendLayers.Length];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new RBM(gpu, rand, saveInfos[i].NumVisible, saveInfos[i].NumHidden,
                    i, saveInfos[i].Weights, saveInfos[i].VisibleActivation, saveInfos[i].HiddenActivation);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
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
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }

            _gpu.Synchronize();
        }

        public CudaDbnD(GPGPU gpu, GPGPURAND rand, ILayerDefinition[] layerSizes)
        {
            _gpu = gpu;
            _rand = rand;
            // ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;


            Machines = new RBM[layerSizes.Length];

            for (int i = 0; i < layerSizes.Length; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits,
                    layerSizes[i].HiddenUnits);

                var rbm = new RBM(gpu, rand, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits,
                    i, layerSizes[i].VisibleActivation, layerSizes[i].HiddenActivation);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public bool Disposed { get; protected set; }

        public int NumMachines
        {
            get { return Machines.Length; }
        }

        public TElement[,] Encode(TElement[,] data)
        {
            return Encode(data, Machines.Length - 1);
        }

        public TElement[,] Encode(TElement[,] data, int maxDepth)
        {
            if (maxDepth < 0)
                return data;

            data = Machines[0].GetHiddenLayer(data);

            for (int i = 0; i < maxDepth; i++)
            {
                data = Machines[i + 1].GetHiddenLayer(data);
            }

            return data;
        }

        public TElement[,] Classify(TElement[,] data, int maxDepth)
        {
            throw new NotImplementedException();
            //if (maxDepth < 0)
            //    return data;

            //data = Machines[0].GetHiddenLayer(data);

            //for (int i = 0; i < maxDepth; i++)
            //{
            //    var m = Machines[i + 1];
            //    data = i + 1 == maxDepth ? m.GetSoftmaxLayer(data) : m.GetHiddenLayer(data);
            //}
            //return data;
            //return  Machines[maxDepth].GetSoftmaxLayer(data);
        }

        public TElement[,] Decode(TElement[,] data)
        {
            return Decode(data, Machines.Length - 1);
        }

        public TElement[,] Decode(TElement[,] data, int maxDepth)
        {
            //data = Machines[maxDepth].GetVisibleLayer(data);

            for (int i = maxDepth; i > -1; i--)
            {
                data = Machines[i].GetVisibleLayer(data);
            }

            //data = Machines[0].GetVisibleLayerLinear(data);

            return data;
        }

        public TElement[,] Reconstruct(TElement[,] data)
        {
            return Reconstruct(data, Machines.Length - 1);
        }

        public TElement[,] Reconstruct(TElement[,] data, int maxDepth)
        {
            TElement[,] hl = Encode(data, maxDepth);
            return Decode(hl, maxDepth);
        }

        public TElement[,] Classify(TElement[,] data, out TElement[,] labels)
        {
            TElement[,] hl = Encode(data, Machines.Length - 2);
            hl = Machines[Machines.Length - 1].Classify(hl, out labels);
            return Decode(hl, Machines.Length - 2);
        }

        public TElement[,] DayDream(int numberOfDreams)
        {
            return DayDream(numberOfDreams, Machines.Length - 1);
        }

        public TElement[,] DayDream(int numberOfDreams, int maxDepth)
        {
            int elems = Machines[0].NumVisibleElements;

            Matrix2D<TElement> dreamRawData;
            _gpu.UniformDistributionBool(_rand, numberOfDreams, elems, out dreamRawData);
            using (dreamRawData)
            {
                //dim3 grid, block;
                //ThreadOptimiser.Instance.GetStrategy(numberOfDreams, elems, out grid, out block);

                //_gpu.Launch(grid, block, Matrix2DCudaF.ToBinaryF, dreamRawData.Matrix);

                //var localRaw = new TElement[numberOfDreams, elems];
                //_gpu.CopyFromDevice(dreamRawData, localRaw);
                TElement[,] ret = Reconstruct(dreamRawData.CopyLocal(), maxDepth);
                return ret;
            }
        }

        /// <summary>
        ///     returns hidden states
        /// </summary>
        /// <param name="data"></param>
        /// <param name="layerIndex"></param>
        /// <param name="error"></param>
        /// <returns></returns>
        public TElement[,] GreedyTrain(TElement[,] data, int layerIndex,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory, out TElement error)
        {
            TElement err = Machines[layerIndex].GreedyTrain(data, exitConditionEvaluatorFactory.Create(layerIndex),
                learningRateFactory.Create(layerIndex));
            RaiseTrainEnd(layerIndex, err);
            error = err;
            return Machines[layerIndex].GetHiddenLayer(data);
        }

        public event EventHandler<EpochEventArgs<TElement>> EpochEnd;

        public event EventHandler<EpochEventArgs<TElement>> TrainEnd;
        //public IExitConditionEvaluatorFactory<TElement> ExitConditionEvaluatorFactory { get; protected set; }

        public IEnumerable<ILayerSaveInfo<TElement>> GetLayerSaveInfos()
        {
            return Machines.Select(restrictedBoltzmannMachineF => restrictedBoltzmannMachineF.GetSaveInfo());
        }

        public TElement GetReconstructionError(TElement[,] srcData, int depth)
        {
            TElement[,] data = Encode(srcData, depth - 1);
            return Machines[depth].CalculateReconstructionError(data);
        }

        /// <summary>
        ///     returns the visible states and labels as out param
        /// </summary>
        /// <param name="data"></param>
        /// <param name="labels"></param>
        /// <param name="layerPosition"></param>
        /// <param name="error"></param>
        /// <param name="labelsPredicted"></param>
        /// <returns></returns>
        public TElement[,] GreedySupervisedTrain(TElement[,] data, TElement[,] labels, int layerPosition,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory, out TElement error,
            out TElement[,] labelsPredicted)
        {
            TElement err = Machines[layerPosition].GreedySupervisedTrain(data, labels,
                exitConditionEvaluatorFactory.Create(layerPosition), learningRateFactory.Create(layerPosition));
            RaiseTrainEnd(layerPosition, err);
            error = err;
            return Machines[layerPosition].Classify(data, out labelsPredicted);
        }

        public Task AsyncGreedyTrain(TElement[,] data, int layerIndex,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement err;
            return Task.Run(
                () => GreedyTrain(data, layerIndex, exitConditionEvaluatorFactory, learningRateFactory, out err));
        }

        public void GreedyTrainAll(TElement[,] visibleData,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            //visibleData = FixInputData(visibleData);

            TElement error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyTrain(visibleData, i, exitConditionEvaluatorFactory, learningRateFactory, out error);
                RaiseTrainEnd(i, error);
            }
        }


        public void UpDownTrainAll(TElement[,] visibleData, int iterations,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;
            for (int i = 0; i < iterations; i++)
            {
                TElement[,] encodedPenultimate = Encode(visibleData, Machines.Length - 2);
                TElement[,] visible = GreedyTrain(encodedPenultimate, Machines.Length - 1, exitConditionEvaluatorFactory,
                    learningRateFactory, out error);

                visible = Machines[Machines.Length - 1].GetVisibleLayer(visible);

                for (int j = Machines.Length - 2; j > -1; j--)
                {
                    Machines[j].DownPass(visible, exitConditionEvaluatorFactory.Create(j), learningRateFactory.Create(j),
                        out error);
                    visible = Machines[j].GetVisibleLayer(visible);
                    //visible = Machines[j].GetVisibleLayer(visible);
                }
            }
        }

        public void UpDownTrainSupervisedAll(TElement[,] visibleData, TElement[,] labels, int iterations,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;
            for (int i = 0; i < iterations; i++)
            {
                TElement[,] encodedPenultimate = Encode(visibleData, Machines.Length - 2);
                TElement[,] labelsPredicted;
                TElement[,] visible = GreedySupervisedTrain(encodedPenultimate, labels, Machines.Length - 1,
                    exitConditionEvaluatorFactory, learningRateFactory, out error,
                    out labelsPredicted);

                for (int j = Machines.Length - 2; j > -1; j--)
                {
                    Machines[j].DownPass(visible, exitConditionEvaluatorFactory.Create(j), learningRateFactory.Create(j),
                        out error);
                    visible = Machines[j].GetVisibleLayer(visible);
                    //visible = Machines[j].GetVisibleLayer(visible);
                }
            }
        }

        public TElement GreedySupervisedTrainAll(TElement[,] visibleData, TElement[,] labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                visibleData = GreedyTrain(visibleData, i, exitConditionEvaluatorFactory, learningRateFactory, out error);
                RaiseTrainEnd(i, error);
            }
            TElement[,] labelsPredicted;
            GreedySupervisedTrain(visibleData, labels, Machines.Length - 1, exitConditionEvaluatorFactory,
                learningRateFactory, out error, out labelsPredicted);
            RaiseTrainEnd(Machines.Length - 1, error);
            return error;
        }

        public TElement GreedyBatchedSupervisedTrainAll(TElement[,] visibleData, TElement[,] labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                visibleData = GreedyBatchedTrain(visibleData, i, batchSize, exitConditionEvaluatorFactory,
                    learningRateFactory, out error);
                RaiseTrainEnd(i, error);
            }
            TElement[,] labelsPredicted;
            //try training supervised layer in one batch
            BatchedSupervisedTrain(visibleData, labels, Machines.Length - 1, batchSize, exitConditionEvaluatorFactory,
                learningRateFactory, out error, out labelsPredicted);
            RaiseTrainEnd(Machines.Length - 1, error);
            return error;
        }

        public Task AsyncGreedyTrainAll(TElement[,] visibleData,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            return Task.Run(() => GreedyTrainAll(visibleData, exitConditionEvaluatorFactory, learningRateFactory));
        }


        public void GreedyTrainLayersFrom(TElement[,] visibleData, int startDepth,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : GreedyTrain(visibleData, i, exitConditionEvaluatorFactory, learningRateFactory, out error);
            }
        }


        public TElement[,] GreedyBatchedTrain(TElement[,] data, int layerPosition, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory, out TElement error)
        {
            TElement err = Machines[layerPosition].GreedyBatchedTrain(data, batchRows,
                exitConditionEvaluatorFactory.Create(layerPosition), learningRateFactory.Create(layerPosition));
            RaiseTrainEnd(layerPosition, err);
            error = err;
            return Machines[layerPosition].GetHiddenLayer(data);
        }

        public Task AsyncGreedyBatchedTrain(TElement[,] data, int layerPosition, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement err;
            return Task.Run(
                () =>
                    GreedyBatchedTrain(data, layerPosition, batchRows, exitConditionEvaluatorFactory,
                        learningRateFactory, out err));
        }

        public void GreedyBatchedTrainAll(TElement[,] visibleData, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyBatchedTrain(visibleData, i, batchRows, exitConditionEvaluatorFactory,
                    learningRateFactory, out error);
                RaiseTrainEnd(i, error);
            }
        }

        public Task AsyncGreedyBatchedTrainAll(TElement[,] visibleData, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            return
                Task.Run(
                    () =>
                        GreedyBatchedTrainAll(visibleData, batchRows, exitConditionEvaluatorFactory, learningRateFactory));
        }

        public void GreedyBatchedTrainLayersFrom(TElement[,] visibleData, int startDepth, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            TElement error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : GreedyBatchedTrain(visibleData, i, batchRows, exitConditionEvaluatorFactory, learningRateFactory,
                        out error);
            }
        }

        public TElement[,] GenerateExamplesByLabel(TElement[,] labels)
        {
            TElement[,] visStates;
            using (
                Matrix2D<TElement> vis = _gpu.AllocateAndSet<TElement>(labels.GetLength(0),
                    Machines[Machines.Length - 1].NumVisibleElements))
            using (Matrix2D<TElement> tmp = _gpu.Upload(labels))
            {
                vis.InsertValuesFrom(0, vis.GetLength(1) - labels.GetLength(1), tmp);
                visStates = vis.CopyLocal();
            }

            TElement[,] data = Machines[Machines.Length - 1].GetHiddenLayer(visStates);
            data = Machines[Machines.Length - 1].GetVisibleLayer(data);

            using (Matrix2D<TElement> m = _gpu.Upload(data))
            using (Matrix2D<TElement> m1 = m.SubMatrix(0, 0, 0, Machines[Machines.Length - 2].NumHiddenElements))
            {
                m.Dispose();
                data = m1.CopyLocal();
                m1.Dispose();

            }

            for (int i = Machines.Length - 2; i > -1; i--)
            {
                data = Machines[i].GetVisibleLayer(data);
            }

            return data;
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

        public TElement[,] BatchedSupervisedTrain(TElement[,] data, TElement[,] labels, int layerPosition, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory,
            out TElement error, out TElement[,] labelsPredicted)
        {
            TElement err = Machines[layerPosition].GreedyBatchedSupervisedTrain(data, labels, batchSize,
                exitConditionEvaluatorFactory.Create(layerPosition), learningRateFactory.Create(layerPosition));
            RaiseTrainEnd(layerPosition, err);
            error = err;
            return Machines[layerPosition].Classify(data, out labelsPredicted);
        }

        private void RaiseTrainEnd(int layer, TElement error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<TElement>
                {
                    Layer = layer,
                    Epoch = -1,
                    Error = error
                });
        }

        private void OnRbm_EpochEnd(object sender, EpochEventArgs<TElement> e)
        {
            RaiseEpochEnd(e);
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
                foreach (RBM restrictedBoltzmannMachineF in Machines)
                {
                    restrictedBoltzmannMachineF.Dispose();
                }
                _rand.Dispose();
                _gpu.FreeAll();
                _gpu.UnloadModules();
                _gpu.Dispose();
            }
        }

        ~CudaDbnD()
        {
            Dispose(false);
        }
    }
}