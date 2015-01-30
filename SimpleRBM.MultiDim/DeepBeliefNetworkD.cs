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
    public class DeepBeliefNetworkD : IDeepBeliefNetwork<double>
    {
        private readonly RestrictedBoltzmannMachineD[] Machines;

        public DeepBeliefNetworkD(DirectoryInfo network, ILearningRateCalculator<double> learningRateCalculator,
            IExitConditionEvaluatorFactory<double> exitConditionExitConditionEvaluatorFactory, ILayerDefinition[] appendLayers = null)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
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
                    saveInfos[i].Weights,
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

                    var rbm = new RestrictedBoltzmannMachineD(
                        appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits,
                        ExitConditionEvaluatorFactory.Create(saveInfos.Count + j,
                            appendLayers[j].VisibleUnits, appendLayers[j].HiddenUnits), learningRateCalculator);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }
        }

        public DeepBeliefNetworkD(ILayerDefinition[] layerSizes, ILearningRateCalculator<double> learningRateCalculator,
            IExitConditionEvaluatorFactory<double> exitConditionExitConditionEvaluatorFactory)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            Machines = new RestrictedBoltzmannMachineD[layerSizes.Length];

            for (int i = 0; i < layerSizes.Length; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits);

                var rbm = new RestrictedBoltzmannMachineD(layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits,
                    exitConditionExitConditionEvaluatorFactory.Create(i, layerSizes[i].VisibleUnits, layerSizes[i].HiddenUnits), learningRateCalculator);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
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

        public double[,] Encode(double[,] data)
        {
            data = Machines[0].GetHiddenLayer(data);

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                data = Machines[i + 1].GetHiddenLayer(data);
            }

            return data;
        }

        public double[,] Decode(double[,] data)
        {
            data = Machines[Machines.Length - 1].GetVisibleLayer(data);

            for (int i = Machines.Length - 1; i > 0; i--)
            {
                data = Machines[i - 1].GetVisibleLayer(data);
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
                Machines[0].NumVisibleElements);

            double[,] ret = Reconstruct(dreamRawData);

            return ret;
        }

        public double[,] GreedyTrain(double[,] data, int layerNumber, out double error)
        {
            double err = Machines[layerNumber].GreedyTrain(data);
            RaiseTrainEnd(err);
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
                RaiseTrainEnd(error);
            }
        }

        public Task AsyncGreedyTrainAll(double[,] visibleData)
        {
            return Task.Run(() => GreedyTrainAll(visibleData));
        }


        public event EventHandler<EpochEventArgs<double>> EpochEnd;

        public event EventHandler<EpochEventArgs<double>> TrainEnd;

        public int NumMachines
        {
            get { return Machines.Length; }
        }

        public IExitConditionEvaluatorFactory<double> ExitConditionEvaluatorFactory { get; protected set; }

        public IEnumerable<ILayerSaveInfo<double>> GetLayerSaveInfos()
        {
            return Machines.Select(a => a.GetSaveInfo());
        }

        private void RaiseTrainEnd(double error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<double>
                {
                    Epoch = -1,
                    Error = error
                });
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


        public double[,] GreedyBatchedTrain(double[,] data, int layerPosition, int batchRows, out double error)
        {
            throw new NotImplementedException();
        }

        public Task AsyncGreedyBatchedTrain(double[,] data, int layerPosition, int batchRows)
        {
            throw new NotImplementedException();
        }

        public void GreedyBatchedTrainAll(double[,] visibleData, int batchRows)
        {
            throw new NotImplementedException();
        }

        public Task AsyncGreedyBatchedTrainAll(double[,] visibleData, int batchRows)
        {
            throw new NotImplementedException();
        }

        public void GreedyBatchedTrainLayersFrom(double[,] visibleData, int startDepth, int batchRows)
        {
            throw new NotImplementedException();
        }
    }
}