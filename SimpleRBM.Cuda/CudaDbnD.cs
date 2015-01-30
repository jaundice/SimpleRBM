using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;

namespace SimpleRBM.Cuda
{
    public class CudaDbnD : IDeepBeliefNetworkExtended<double>, IDisposable
    {
        private readonly CudaRbmD[] Machines;

        public int NumMachines
        {
            get { return Machines.Length; }
        }
        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;


        public CudaDbnD(GPGPU gpu, GPGPURAND rand, DirectoryInfo network, ILearningRateCalculator<double> learningRate,
            IExitConditionEvaluatorFactory<double> exitConditionExitConditionEvaluatorFactory, ILayerDefinition[] appendLayers = null)
        {
            _gpu = gpu;
            _rand = rand;
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            List<LayerSaveInfoD> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoD(a.FullName)).ToList();

            appendLayers = appendLayers ?? new ILayerDefinition[0];
            Machines =
                new CudaRbmD[saveInfos.Count() + appendLayers.Length];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new CudaRbmD(gpu, rand, saveInfos[i].NumVisible, saveInfos[i].NumHidden,
                    i, saveInfos[i].Weights,
                    ExitConditionEvaluatorFactory.Create(i, saveInfos[i].NumVisible, saveInfos[i].NumHidden),
                    learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }

            if (appendLayers.Length > 0)
            {
                for (int j = 0; j < appendLayers.Length; j++)
                {
                    Console.WriteLine("Appending Layer {0}: {1}x{2}", j + saveInfos.Count,
                        appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits);

                    var rbm = new CudaRbmD(gpu, rand,
                        appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits,
                       saveInfos.Count + j + 1, ExitConditionEvaluatorFactory.Create(saveInfos.Count + j,
                            appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits), learningRate);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }

            _gpu.Synchronize();
        }

        public CudaDbnD(GPGPU gpu, GPGPURAND rand, ILayerDefinition[] layerSizes, ILearningRateCalculator<double> learningRate,
            IExitConditionEvaluatorFactory<double> exitConditionExitConditionEvaluatorFactory)
        {
            _gpu = gpu;
            _rand = rand;
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;


            Machines = new CudaRbmD[layerSizes.Length];

            for (int i = 0; i < layerSizes.Length; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits);

                var rbm = new CudaRbmD(gpu, rand, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits,
                    i, ExitConditionEvaluatorFactory.Create(i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits), learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public bool Disposed { get; protected set; }

        public double[,] Encode(double[,] data)
        {
            return Encode(data, Machines.Length - 1);
        }
        public double[,] Encode(double[,] data, int maxDepth)
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

        public double[,] Decode(double[,] data)
        {
            return Decode(data, Machines.Length - 1);
        }
        public double[,] Decode(double[,] data, int maxDepth)
        {
            data = Machines[maxDepth].GetVisibleLayer(data);

            for (int i = maxDepth; i > 0; i--)
            {
                data = Machines[i - 1].GetVisibleLayer(data);
            }

            return data;
        }

        public double[,] Reconstruct(double[,] data)
        {
            return Reconstruct(data, Machines.Length - 1);
        }
        public double[,] Reconstruct(double[,] data, int maxDepth)
        {
            double[,] hl = Encode(data, maxDepth);
            return Decode(hl, maxDepth);
        }

        public double[,] DayDream(int numberOfDreams)
        {
            return DayDream(numberOfDreams, Machines.Length - 1);
        }
        public double[,] DayDream(int numberOfDreams, int maxDepth)
        {
            int elems = Machines[0].NumVisibleElements;
            using (
                Matrix2D<double> dreamRawData = CudaRbmD.UniformDistribution(_gpu, _rand,
                    numberOfDreams, elems))
            {
                dim3 grid, block;
                ThreadOptimiser.Instance.GetStrategy(numberOfDreams, elems, out grid, out block);

                _gpu.Launch(grid, block, Matrix2DCudaD.ToBinaryD, dreamRawData.Matrix);

                var localRaw = new double[numberOfDreams, elems];
                _gpu.CopyFromDevice(dreamRawData, localRaw);
                double[,] ret = Reconstruct(localRaw, maxDepth);
                return ret;
            }
        }

        public double[,] GreedyTrain(double[,] data, int layerNumber, out double error)
        {
            double err = Machines[layerNumber].GreedyTrain(data);
            RaiseTrainEnd(layerNumber, err);
            error = err;
            return Machines[layerNumber].GetHiddenLayer(data);
        }

        public Task AsyncGreedyTrain(double[,] data, int layerPosition)
        {
            double err;
            return Task.Run(
                () => GreedyTrain(data, layerPosition, out err));
        }

        public void GreedyTrainAll(double[,] visibleData)
        {
            double error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyTrain(visibleData, i, out error);
                RaiseTrainEnd(i, error);
            }
        }

        public Task AsyncGreedyTrainAll(double[,] visibleData)
        {
            return Task.Run(() => GreedyTrainAll(visibleData));
        }


        public event EventHandler<EpochEventArgs<double>> EpochEnd;

        public event EventHandler<EpochEventArgs<double>> TrainEnd;
        public IExitConditionEvaluatorFactory<double> ExitConditionEvaluatorFactory { get; protected set; }

        public IEnumerable<ILayerSaveInfo<double>> GetLayerSaveInfos()
        {
            return Machines.Select(restrictedBoltzmannMachineF => restrictedBoltzmannMachineF.GetSaveInfo());
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

        public void GreedyTrainLayersFrom(double[,] visibleData, int startDepth)
        {
            double error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : GreedyTrain(visibleData, i, out error);
            }
        }

        private void RaiseTrainEnd(int layer, double error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<double>
                {
                    Layer = layer,
                    Epoch = -1,
                    Error = error
                });
        }

        private void OnRbm_EpochEnd(object sender, EpochEventArgs<double> e)
        {
            RaiseEpochEnd(e);
        }

        private void RaiseEpochEnd(EpochEventArgs<double> e)
        {
            if (EpochEnd != null)
                EpochEnd(this, e);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (CudaRbmD restrictedBoltzmannMachineF in Machines)
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


        public double[,] GreedyBatchedTrain(double[,] data, int layerPosition, int batchRows, out double error)
        {
            double err = Machines[layerPosition].GreedyBatchedTrain(data, batchRows);
            RaiseTrainEnd(layerPosition, err);
            error = err;
            return Machines[layerPosition].GetHiddenLayer(data);
        }

        public Task AsyncGreedyBatchedTrain(double[,] data, int layerPosition, int batchRows)
        {
            double err;
            return Task.Run(
                () => GreedyBatchedTrain(data, layerPosition, batchRows, out err));
        }

        public void GreedyBatchedTrainAll(double[,] visibleData, int batchRows)
        {
            double error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyBatchedTrain(visibleData, i, batchRows, out error);
                RaiseTrainEnd(i, error);
            }
        }

        public Task AsyncGreedyBatchedTrainAll(double[,] visibleData, int batchRows)
        {
            return Task.Run(() => GreedyBatchedTrainAll(visibleData, batchRows));
        }

        public void GreedyBatchedTrainLayersFrom(double[,] visibleData, int startDepth, int batchRows)
        {
            double error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : GreedyBatchedTrain(visibleData, i, batchRows, out error);
            }
        }


        public double GetReconstructionError(double[,] srcData, int depth)
        {
            var data = Encode(srcData, depth - 1);
            return Machines[depth].CalculateReconstructionError(data);
        }

        public double[,] Classify(double[,] data, int maxDepth)
        {
            throw new NotImplementedException();
        }


        public double GreedySupervisedTrainAll(double[,] srcData, double[,] labels)
        {
            throw new NotImplementedException();
        }

        public double[,] Classify(double[,] data, out double[,] labels)
        {
            throw new NotImplementedException();
        }


        public double[,] GreedySupervisedTrain(double[,] data, double[,] labels, int layerPosition, out double error, out double[,] labelsPredicted)
        {
            throw new NotImplementedException();
        }


        public double GreedyBatchedSupervisedTrainAll(double[,] srcData, double[,] labels, int batchSize)
        {
            throw new NotImplementedException();
        }


        public void UpDownTrainAll(double[,] visibleData, int iterations, int epochsPerMachine, double learningRate)
        {
            throw new NotImplementedException();
        }


        public void UpDownTrainSupervisedAll(double[,] visibleData, double[,] labels, int iterations, int epochsPerMachine, double learningRate)
        {
            throw new NotImplementedException();
        }
    }
}