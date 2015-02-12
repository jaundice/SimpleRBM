using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;
using TElement = System.Single;

namespace SimpleRBM.MultiDim
{
    public class DeepBeliefNetworkF : IDeepBeliefNetwork<TElement>
    {
        private readonly RestrictedBoltzmannMachineF[] Machines;

        public DeepBeliefNetworkF(DirectoryInfo network, ILayerDefinition[] appendLayers = null)
        {
            List<LayerSaveInfoF> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoF(a.FullName)).ToList();

            appendLayers = appendLayers ?? new ILayerDefinition[0];
            Machines =
                new RestrictedBoltzmannMachineF[saveInfos.Count() + (appendLayers.Length == 0 ? 0 : appendLayers.Length)
                    ];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new RestrictedBoltzmannMachineF(saveInfos[i].NumVisible, saveInfos[i].NumHidden,
                    saveInfos[i].Weights, saveInfos[i].VisibleActivation, saveInfos[i].HiddenActivation);
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
                        appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits, appendLayers[j].VisibleActivation, appendLayers[j].HiddenActivation);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }
        }

        public DeepBeliefNetworkF(ILayerDefinition[] layerSizes)
        {
            Machines = new RestrictedBoltzmannMachineF[layerSizes.Length];

            for (int i = 0; i < layerSizes.Length; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits,
                    layerSizes[i].HiddenUnits);

                var rbm = new RestrictedBoltzmannMachineF(layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits, layerSizes[i].VisibleActivation, layerSizes[i].HiddenActivation);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public void GreedyTrainLayersFrom(TElement[,] visibleData, int startDepth,
            IExitConditionEvaluatorFactory<TElement> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateCalculatorFactory)
        {
            TElement error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : GreedyTrain(visibleData, i, exitEvaluatorFactory, learningRateCalculatorFactory, out error);
            }
        }

        public TElement[,] Encode(TElement[,] data)
        {
            data = Machines[0].GetHiddenLayer(data);

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                data = Machines[i + 1].GetHiddenLayer(data);
            }

            return data;
        }

        public TElement[,] Decode(TElement[,] data)
        {
            data = Machines[Machines.Length - 1].GetVisibleLayer(data);

            for (int i = Machines.Length - 1; i > 0; i--)
            {
                data = Machines[i - 1].GetVisibleLayer(data);
            }

            return data;
        }

        public TElement[,] Reconstruct(TElement[,] data)
        {
            TElement[,] hl = Encode(data);
            return Decode(hl);
        }

        public TElement[,] DayDream(int numberOfDreams)
        {
            TElement[,] dreamRawData = Distributions.UniformRandromMatrixBoolF(numberOfDreams,
                Machines[0].NumVisibleElements);

            TElement[,] ret = Reconstruct(dreamRawData);

            return ret;
        }

        public TElement[,] GreedyTrain(TElement[,] data, int layerIndex,
            IExitConditionEvaluatorFactory<TElement> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateCalculatorFactory, out TElement error)
        {
            TElement err = Machines[layerIndex].GreedyTrain(data, exitEvaluatorFactory.Create(layerIndex),
                learningRateCalculatorFactory.Create(layerIndex));
            RaiseTrainEnd(err);
            error = err;
            return Machines[layerIndex].GetHiddenLayer(data);
        }

        public Task AsyncGreedyTrain(TElement[,] data, int layerIndex,
            IExitConditionEvaluatorFactory<TElement> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateCalculatorFactory)
        {
            TElement err;
            return Task.Run(
                () => GreedyTrain(data, layerIndex, exitEvaluatorFactory, learningRateCalculatorFactory, out err));
        }

        public void GreedyTrainAll(TElement[,] visibleData, IExitConditionEvaluatorFactory<TElement> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateCalculatorFactory)
        {
            TElement error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = GreedyTrain(visibleData, i, exitEvaluatorFactory, learningRateCalculatorFactory, out error);
                RaiseTrainEnd(error);
            }
        }

        public Task AsyncGreedyTrainAll(TElement[,] visibleData,
            IExitConditionEvaluatorFactory<TElement> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateCalculatorFactory)
        {
            return Task.Run(() => GreedyTrainAll(visibleData, exitEvaluatorFactory, learningRateCalculatorFactory));
        }


        public event EventHandler<EpochEventArgs<TElement>> EpochEnd;

        public event EventHandler<EpochEventArgs<TElement>> TrainEnd;

        public int NumMachines
        {
            get { return Machines.Length; }
        }

        public IEnumerable<ILayerSaveInfo<TElement>> GetLayerSaveInfos()
        {
            return Machines.Select(a => a.GetSaveInfo());
        }


        public TElement[,] GreedyBatchedTrain(TElement[,] data, int layerPosition, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory, out TElement error)
        {
            throw new NotImplementedException();
        }

        public Task AsyncGreedyBatchedTrain(TElement[,] data, int layerPosition, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            throw new NotImplementedException();
        }

        public void GreedyBatchedTrainAll(TElement[,] visibleData, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            throw new NotImplementedException();
        }

        public Task AsyncGreedyBatchedTrainAll(TElement[,] visibleData, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            throw new NotImplementedException();
        }

        public void GreedyBatchedTrainLayersFrom(TElement[,] visibleData, int startDepth, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory)
        {
            throw new NotImplementedException();
        }

        private void RaiseTrainEnd(TElement error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<TElement>
                {
                    Epoch = -1,
                    Error = error
                });
        }

        private void OnRbm_EpochEnd(object sender, EpochEventArgs<TElement> e)
        {
            RaiseEpochEnd(e.Epoch, e.Error);
        }

        private void RaiseEpochEnd(int epoch, TElement error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<TElement>
                {
                    Epoch = epoch,
                    Error = error
                });
        }
    }
}