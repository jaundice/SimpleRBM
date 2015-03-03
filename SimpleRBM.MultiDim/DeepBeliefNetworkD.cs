using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;

namespace SimpleRBM.MultiDim
{
    public class DeepBeliefNetworkD : IDeepBeliefNetwork<double>
    {
        private readonly RestrictedBoltzmannMachineD[] Machines;

        public DeepBeliefNetworkD(DirectoryInfo network, ILayerDefinition[] appendLayers = null)
        {
            List<LayerSaveInfoD> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoD(a.FullName)).ToList();

            appendLayers = appendLayers ?? new ILayerDefinition[0];
            Machines =
                new RestrictedBoltzmannMachineD[saveInfos.Count() + (appendLayers.Length == 0 ? 0 : appendLayers.Length)
                    ];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new RestrictedBoltzmannMachineD(saveInfos[i].NumVisible, saveInfos[i].NumHidden,
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

                    var rbm = new RestrictedBoltzmannMachineD(
                        appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits, appendLayers[j].VisibleActivation, appendLayers[j].HiddenActivation);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }
        }

        public DeepBeliefNetworkD(ILayerDefinition[] layerSizes)
        {
            Machines = new RestrictedBoltzmannMachineD[layerSizes.Length];

            for (int i = 0; i < layerSizes.Length; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits,
                    layerSizes[i].HiddenUnits);

                var rbm = new RestrictedBoltzmannMachineD(layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits, layerSizes[i].VisibleActivation, layerSizes[i].HiddenActivation);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public void GreedyTrainLayersFrom(double[,] visibleData, int startDepth,
            IExitConditionEvaluatorFactory<double> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<double> learningRateCalculatorFactory, CancellationToken cancelToken)
        {
            for (int i = 0; i < Machines.Length; i++)
            {
                cancelToken.ThrowIfCancellationRequested();
                visibleData = i < startDepth
                    ? Machines[i].Encode(visibleData)
                    : GreedyTrain(visibleData, i, exitEvaluatorFactory, learningRateCalculatorFactory, cancelToken);
            }
        }

        public double[,] Encode(double[,] data)
        {
            data = Machines[0].Encode(data);

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                data = Machines[i + 1].Encode(data);
            }

            return data;
        }

        public double[,] Decode(double[,] data)
        {
            data = Machines[Machines.Length - 1].Decode(data);

            for (int i = Machines.Length - 1; i > 0; i--)
            {
                data = Machines[i - 1].Decode(data);
            }

            return data;
        }

        public double[,] Reconstruct(double[,] data)
        {
            double[,] hl = Encode(data);
            return Decode(hl);
        }

        public double[,] DayDream(int numberOfDreams)
        {
            double[,] dreamRawData = Distributions.UniformRandromMatrixBoolD(numberOfDreams,
                Machines[0].NumVisibleNeurons);

            double[,] ret = Reconstruct(dreamRawData);

            return ret;
        }

        public double[,] GreedyTrain(double[,] data, int layerIndex,
            IExitConditionEvaluatorFactory<double> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<double> learningRateCalculatorFactory, CancellationToken cancelToken)
        {
            Machines[layerIndex].GreedyTrain(data, exitEvaluatorFactory.Create(layerIndex),
                 learningRateCalculatorFactory.Create(layerIndex), cancelToken);

            return Machines[layerIndex].Encode(data);
        }

        public Task AsyncGreedyTrain(double[,] data, int layerIndex,
            IExitConditionEvaluatorFactory<double> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<double> learningRateCalculatorFactory, CancellationToken cancelToken)
        {
            return Task.Run(
                () => GreedyTrain(data, layerIndex, exitEvaluatorFactory, learningRateCalculatorFactory, cancelToken), cancelToken);
        }

        public void GreedyTrainAll(double[,] visibleData, IExitConditionEvaluatorFactory<double> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<double> learningRateCalculatorFactory, CancellationToken cancelToken)
        {
            double error;

            for (int i = 0; i < Machines.Length; i++)
            {
                cancelToken.ThrowIfCancellationRequested();
                visibleData = GreedyTrain(visibleData, i, exitEvaluatorFactory, learningRateCalculatorFactory, cancelToken);
            }
        }

        public Task AsyncGreedyTrainAll(double[,] visibleData,
            IExitConditionEvaluatorFactory<double> exitEvaluatorFactory,
            ILearningRateCalculatorFactory<double> learningRateCalculatorFactory, CancellationToken cancelToken)
        {
            return Task.Run(() => GreedyTrainAll(visibleData, exitEvaluatorFactory, learningRateCalculatorFactory, cancelToken), cancelToken);
        }


        public event EventHandler<EpochEventArgs<double>> EpochEnd;

        public event EventHandler<EpochEventArgs<double>> TrainEnd;

        public int NumMachines
        {
            get { return Machines.Length; }
        }

        public IEnumerable<ILayerSaveInfo<double>> GetLayerSaveInfos()
        {
            return Machines.Select(a => a.GetSaveInfo());
        }


        public void GreedyBatchedTrainAll(double[,] visibleData, int batchRows,
            IExitConditionEvaluatorFactory<double> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<double> learningRateFactory, CancellationToken cancelToken)
        {
            throw new NotImplementedException();
        }

        private void OnRbm_EpochEnd(object sender, EpochEventArgs<double> e)
        {
            RaiseEpochEnd(e.Epoch, e.Error);
        }

        private void RaiseEpochEnd(int epoch, double error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<double>
                {
                    Epoch = epoch,
                    Error = error
                });
        }
    }
}