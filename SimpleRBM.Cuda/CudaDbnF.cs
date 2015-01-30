using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Mono.CSharp;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;

namespace SimpleRBM.Cuda
{
    public class CudaDbnF : IDeepBeliefNetworkExtended<float>, IDisposable
    {
        private readonly CudaRbmF[] Machines;

        public int NumMachines
        {
            get { return Machines.Length; }
        }
        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;


        public CudaDbnF(GPGPU gpu, GPGPURAND rand, DirectoryInfo network, ILearningRateCalculator<float> learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory, ILayerDefinition[] appendLayers = null)
        {
            _gpu = gpu;
            _rand = rand;
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            List<LayerSaveInfoF> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoF(a.FullName)).ToList();

            appendLayers = appendLayers ?? new ILayerDefinition[0];
            Machines =
                new CudaRbmF[saveInfos.Count() + appendLayers.Length];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new CudaRbmF(gpu, rand, saveInfos[i].NumVisible, saveInfos[i].NumHidden,
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

                    var rbm = new CudaRbmF(gpu, rand,
                        appendLayers[j].VisibleUnits, appendLayers[j + 1].HiddenUnits,
                        saveInfos.Count + j, ExitConditionEvaluatorFactory.Create(saveInfos.Count + j,
                           appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits), learningRate);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }

            _gpu.Synchronize();
        }

        public CudaDbnF(GPGPU gpu, GPGPURAND rand, ILayerDefinition[] layerSizes, ILearningRateCalculator<float> learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory)
        {
            _gpu = gpu;
            _rand = rand;
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;


            Machines = new CudaRbmF[layerSizes.Length];

            for (int i = 0; i < layerSizes.Length; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits);

                var rbm = new CudaRbmF(gpu, rand, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits,
                    i, ExitConditionEvaluatorFactory.Create(i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits), learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public bool Disposed { get; protected set; }

        public float[,] Encode(float[,] data)
        {
            return Encode(data, Machines.Length - 1);
        }
        public float[,] Encode(float[,] data, int maxDepth)
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

        public float[,] Classify(float[,] data, int maxDepth)
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

        public float[,] Decode(float[,] data)
        {
            return Decode(data, Machines.Length - 1);
        }
        public float[,] Decode(float[,] data, int maxDepth)
        {

            //data = Machines[maxDepth].GetVisibleLayer(data);

            for (int i = maxDepth; i > 0; i--)
            {
                data = Machines[i].GetVisibleLayer(data);
            }

            data = Machines[0].GetVisibleLayerLinear(data);

            return data;
        }

        public float[,] Reconstruct(float[,] data)
        {
            return Reconstruct(data, Machines.Length - 1);
        }
        public float[,] Reconstruct(float[,] data, int maxDepth)
        {
            float[,] hl = Encode(data, maxDepth);
            return Decode(hl, maxDepth);
        }

        public float[,] Classify(float[,] data, out float[,] labels)
        {
            float[,] hl = Encode(data, Machines.Length - 2);
            hl = Machines[Machines.Length - 1].Classify(hl, out labels);
            return Decode(hl, Machines.Length - 2);
        }

        public float[,] DayDream(int numberOfDreams)
        {
            return DayDream(numberOfDreams, Machines.Length - 1);
        }
        public float[,] DayDream(int numberOfDreams, int maxDepth)
        {
            int elems = Machines[0].NumVisibleElements;
            using (
                Matrix2D<float> dreamRawData = CudaRbmF.UniformDistributionBool(_gpu, _rand,
                    numberOfDreams, elems))
            {
                //dim3 grid, block;
                //ThreadOptimiser.Instance.GetStrategy(numberOfDreams, elems, out grid, out block);

                //_gpu.Launch(grid, block, Matrix2DCudaF.ToBinaryF, dreamRawData.Matrix);

                //var localRaw = new float[numberOfDreams, elems];
                //_gpu.CopyFromDevice(dreamRawData, localRaw);
                float[,] ret = Reconstruct(dreamRawData.CopyLocal(), maxDepth);
                return ret;
            }
        }

        /// <summary>
        /// returns hidden states
        /// </summary>
        /// <param name="data"></param>
        /// <param name="layerNumber"></param>
        /// <param name="error"></param>
        /// <returns></returns>
        public float[,] GreedyTrain(float[,] data, int layerNumber, out float error)
        {
            float err = Machines[layerNumber].GreedyTrain(data);
            RaiseTrainEnd(layerNumber, err);
            error = err;
            return Machines[layerNumber].GetHiddenLayer(data);
        }
        /// <summary>
        /// returns the visible states and labels as out param
        /// </summary>
        /// <param name="data"></param>
        /// <param name="labels"></param>
        /// <param name="layerPosition"></param>
        /// <param name="error"></param>
        /// <param name="labelsPredicted"></param>
        /// <returns></returns>
        public float[,] GreedySupervisedTrain(float[,] data, float[,] labels, int layerPosition, out float error, out float[,] labelsPredicted)
        {
            float err = Machines[layerPosition].GreedySupervisedTrain(data, labels);
            RaiseTrainEnd(layerPosition, err);
            error = err;
            return Machines[layerPosition].Classify(data, out labelsPredicted);
        }

        public float[,] BatchedSupervisedTrain(float[,] data, float[,] labels, int layerPosition, int batchSize, out float error, out float[,] labelsPredicted)
        {
            float err = Machines[layerPosition].GreedyBatchedSupervisedTrain(data, labels, batchSize);
            RaiseTrainEnd(layerPosition, err);
            error = err;
            return Machines[layerPosition].Classify(data, out labelsPredicted);
        }

        public Task AsyncGreedyTrain(float[,] data, int layerPosition)
        {
            float err;
            return Task.Run(
                () => GreedyTrain(data, layerPosition, out err));
        }

        public void GreedyTrainAll(float[,] visibleData)
        {
            //visibleData = FixInputData(visibleData);

            float error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyTrain(visibleData, i, out error);
                RaiseTrainEnd(i, error);
            }
        }


        public void UpDownTrainAll(float[,] visibleData, int iterations, int epochsPerMachine, float learningRate)
        {
            float error;
            for (int i = 0; i < iterations; i++)
            {
                var encodedPenultimate = Encode(visibleData, Machines.Length - 2);
                var visible = GreedyTrain(encodedPenultimate, Machines.Length - 1, out error);

                visible = Machines[Machines.Length - 1].GetVisibleLayer(visible);

                for (var j = Machines.Length - 2; j > -1; j--)
                {
                    Machines[j].DownPass(visible, epochsPerMachine, learningRate, out error);
                    visible = j == 0 ? Machines[j].GetVisibleLayerLinear(visible) : Machines[j].GetVisibleLayer(visible);
                    //visible = Machines[j].GetVisibleLayer(visible);
                }

            }

        }

        public void UpDownTrainSupervisedAll(float[,] visibleData, float[,] labels, int iterations, int epochsPerMachine, float learningRate)
        {
            float error;
            for (int i = 0; i < iterations; i++)
            {
                var encodedPenultimate = Encode(visibleData, Machines.Length - 2);
                float[,] labelsPredicted;
                var visible = GreedySupervisedTrain(encodedPenultimate, labels, Machines.Length - 1, out error, out labelsPredicted);

                for (var j = Machines.Length - 2; j > -1; j--)
                {
                    Machines[j].DownPass(visible, epochsPerMachine, learningRate, out error);
                    visible = j == 0 ? Machines[j].GetVisibleLayerLinear(visible) : Machines[j].GetVisibleLayer(visible);
                    //visible = Machines[j].GetVisibleLayer(visible);
                }

            }

        }

        public float GreedySupervisedTrainAll(float[,] visibleData, float[,] labels)
        {

            float error;

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                visibleData = GreedyTrain(visibleData, i, out error);
                RaiseTrainEnd(i, error);
            }
            float[,] labelsPredicted;
            GreedySupervisedTrain(visibleData, labels, Machines.Length - 1, out error, out labelsPredicted);
            RaiseTrainEnd(Machines.Length - 1, error);
            return error;
        }

        public float GreedyBatchedSupervisedTrainAll(float[,] visibleData, float[,] labels, int batchSize)
        {

            float error;

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                visibleData = GreedyBatchedTrain(visibleData, i, batchSize, out error);
                RaiseTrainEnd(i, error);
            }
            float[,] labelsPredicted;
            //try training supervised layer in one batch
            BatchedSupervisedTrain(visibleData, labels, Machines.Length - 1, batchSize, out error, out labelsPredicted);
            RaiseTrainEnd(Machines.Length - 1, error);
            return error;
        }

        public Task AsyncGreedyTrainAll(float[,] visibleData)
        {
            return Task.Run(() => GreedyTrainAll(visibleData));
        }


        public event EventHandler<EpochEventArgs<float>> EpochEnd;

        public event EventHandler<EpochEventArgs<float>> TrainEnd;
        public IExitConditionEvaluatorFactory<float> ExitConditionEvaluatorFactory { get; protected set; }

        public IEnumerable<ILayerSaveInfo<float>> GetLayerSaveInfos()
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

        public void GreedyTrainLayersFrom(float[,] visibleData, int startDepth)
        {
            float error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : GreedyTrain(visibleData, i, out error);
            }
        }

        private void RaiseTrainEnd(int layer, float error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<float>
                {
                    Layer = layer,
                    Epoch = -1,
                    Error = error
                });
        }

        private void OnRbm_EpochEnd(object sender, EpochEventArgs<float> e)
        {
            RaiseEpochEnd(e);
        }

        private void RaiseEpochEnd(EpochEventArgs<float> e)
        {
            if (EpochEnd != null)
                EpochEnd(this, e);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (CudaRbmF restrictedBoltzmannMachineF in Machines)
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


        public float[,] GreedyBatchedTrain(float[,] data, int layerPosition, int batchRows, out float error)
        {


            float err = Machines[layerPosition].GreedyBatchedTrain(data, batchRows);
            RaiseTrainEnd(layerPosition, err);
            error = err;
            return Machines[layerPosition].GetHiddenLayer(data);
        }

        public Task AsyncGreedyBatchedTrain(float[,] data, int layerPosition, int batchRows)
        {
            float err;
            return Task.Run(
                () => GreedyBatchedTrain(data, layerPosition, batchRows, out err));

        }

        public void GreedyBatchedTrainAll(float[,] visibleData, int batchRows)
        {

            float error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyBatchedTrain(visibleData, i, batchRows, out error);
                RaiseTrainEnd(i, error);
            }
        }

        public Task AsyncGreedyBatchedTrainAll(float[,] visibleData, int batchRows)
        {
            return Task.Run(() => GreedyBatchedTrainAll(visibleData, batchRows));
        }

        public void GreedyBatchedTrainLayersFrom(float[,] visibleData, int startDepth, int batchRows)
        {

            float error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : GreedyBatchedTrain(visibleData, i, batchRows, out error);
            }
        }


        public float GetReconstructionError(float[,] srcData, int depth)
        {
            var data = Encode(srcData, depth - 1);
            return Machines[depth].CalculateReconstructionError(data);
        }

    }
}