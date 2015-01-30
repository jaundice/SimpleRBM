using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;

namespace SimpleRBM.MultiDim
{
    public class DeepBeliefNetworkF : IDeepBeliefNetworkExtended<float>
    {
        private readonly RestrictedBoltzmannMachineF[] Machines;

        public DeepBeliefNetworkF(DirectoryInfo network, ILearningRateCalculator<float> learningRateCalculator,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory, ILayerDefinition[] appendLayers = null)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            List<LayerSaveInfoF> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoF(a.FullName)).ToList();

            appendLayers = appendLayers ?? new ILayerDefinition[0];
            Machines =
                new RestrictedBoltzmannMachineF[saveInfos.Count() + appendLayers.Length];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new RestrictedBoltzmannMachineF(saveInfos[i].NumVisible, saveInfos[i].NumHidden,
                    i, saveInfos[i].Weights,
                    ExitConditionEvaluatorFactory.Create(i, saveInfos[i].NumVisible, saveInfos[i].NumHidden),
                    learningRateCalculator);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }

            if (appendLayers.Length > 0)
            {
                for (int j = 0; j < appendLayers.Length; j++)
                {
                    Console.WriteLine("Appending Layer {0}: {1}x{2}", j + saveInfos.Count,
                       appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits);

                    var rbm = new RestrictedBoltzmannMachineF(
                        appendLayers[j].VisibleUnits, appendLayers[j ].HiddenUnits,
                        saveInfos.Count + j + 1, ExitConditionEvaluatorFactory.Create(saveInfos.Count + j,
                            appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits), learningRateCalculator);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }
        }

        public DeepBeliefNetworkF(ILayerDefinition[] layerSizes, ILearningRateCalculator<float> learningRateCalculator,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            Machines = new RestrictedBoltzmannMachineF[layerSizes.Length];

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits);

                var rbm = new RestrictedBoltzmannMachineF(layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits,
                    i, exitConditionExitConditionEvaluatorFactory.Create(i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits), learningRateCalculator);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
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

        public float[,] Encode(float[,] data)
        {
            return Encode(data, Machines.Length - 1);
        }
        public float[,] Encode(float[,] data, int maxDepth)
        {
            data = Machines[0].GetHiddenLayer(data);

            for (int i = 0; i < maxDepth; i++)
            {
                data = Machines[i + 1].GetHiddenLayer(data);
            }

            return data;
        }

        public float[,] Decode(float[,] data)
        {
            return Decode(data, Machines.Length - 1);
        }
        public float[,] Decode(float[,] data, int maxDepth)
        {
            data = Machines[maxDepth].GetVisibleLayer(data);

            for (int i = maxDepth; i > 0; i--)
            {
                data = Machines[i - 1].GetVisibleLayer(data);
            }

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

        public float[,] DayDream(int numberOfDreams)
        {
            return DayDream(numberOfDreams, Machines.Length - 1);
        }
        public float[,] DayDream(int numberOfDreams, int maxDepth)
        {
            float[,] dreamRawData = Distributions.UniformRandromMatrixBoolF(numberOfDreams,
                Machines[0].NumVisibleElements);

            float[,] ret = Reconstruct(dreamRawData, maxDepth);

            return ret;
        }

        public float[,] GreedyTrain(float[,] data, int layerNumber, out float error)
        {
            float err = Machines[layerNumber].GreedyTrain(data);
            RaiseTrainEnd(err);
            error = err;
            return Machines[layerNumber].GetHiddenLayer(data);
        }

        public Task AsyncGreedyTrain(float[,] data, int layerPosition)
        {
            float err;
            return Task.Run(
                () => GreedyTrain(data, layerPosition, out err));
        }

        public void GreedyTrainAll(float[,] visibleData)
        {
            float error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyTrain(visibleData, i, out error);
                RaiseTrainEnd(error);
            }
        }

        public Task AsyncGreedyTrainAll(float[,] visibleData)
        {
            return Task.Run(() => GreedyTrainAll(visibleData));
        }


        public event EventHandler<EpochEventArgs<float>> EpochEnd;

        public event EventHandler<EpochEventArgs<float>> TrainEnd;

        public int NumMachines
        {
            get { return Machines.Length; }
        }

        public IExitConditionEvaluatorFactory<float> ExitConditionEvaluatorFactory { get; protected set; }

        public IEnumerable<ILayerSaveInfo<float>> GetLayerSaveInfos()
        {
            return Machines.Select(a => a.GetSaveInfo());
        }

        private void RaiseTrainEnd(float error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<float>
                {
                    Epoch = -1,
                    Error = error
                });
        }

        private void OnRbm_EpochEnd(object sender, EpochEventArgs<float> e)
        {
            RaiseEpochEnd(e.Epoch, e.Error);
        }

        private void RaiseEpochEnd(int epoch, float error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<float>
                {
                    Epoch = epoch,
                    Error = error
                });
        }


        public float[,] GreedyBatchedTrain(float[,] data, int layerPosition, int batchRows, out float error)
        {
            throw new NotImplementedException();
        }

        public Task AsyncGreedyBatchedTrain(float[,] data, int layerPosition, int batchRows)
        {
            throw new NotImplementedException();
        }

        public void GreedyBatchedTrainAll(float[,] visibleData, int batchRows)
        {
            throw new NotImplementedException();
        }

        public Task AsyncGreedyBatchedTrainAll(float[,] visibleData, int batchRows)
        {
            throw new NotImplementedException();
        }

        public void GreedyBatchedTrainLayersFrom(float[,] visibleData, int startDepth, int batchRows)
        {
            throw new NotImplementedException();
        }


        public float GetReconstructionError(float[,] srcData, int depth)
        {
            throw new NotImplementedException();
        }

        public float[,] Classify(float[,] data, int maxDepth)
        {
            throw new NotImplementedException();
        }


        public float GreedySupervisedTrainAll(float[,] srcData, float[,] labels)
        {
            throw new NotImplementedException();
        }

        public float[,] Classify(float[,] data, out float[,] labels)
        {
            throw new NotImplementedException();
        }


        public float[,] GreedySupervisedTrain(float[,] data, float[,] labels, int layerPosition, out float error, out float[,] labelsPredicted)
        {
            throw new NotImplementedException();
        }


        public float GreedyBatchedSupervisedTrainAll(float[,] srcData, float[,] labels, int batchSize)
        {
            throw new NotImplementedException();
        }


        public void UpDownTrainAll(float[,] visibleData, int iterations, int epochsPerMachine, float learningRate)
        {
            throw new NotImplementedException();
        }


        public void UpDownTrainSupervisedAll(float[,] visibleData, float[,] labels, int iterations, int epochsPerMachine, float learningRate)
        {
            throw new NotImplementedException();
        }
    }
}