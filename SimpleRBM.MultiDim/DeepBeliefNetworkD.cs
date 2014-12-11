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

        public DeepBeliefNetworkD(DirectoryInfo network, double learningRate,
            IExitConditionEvaluatorFactory<double> exitConditionExitConditionEvaluatorFactory, int[] appendLayers = null)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            List<LayerSaveInfoD> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoD(a.FullName)).ToList();

            appendLayers = appendLayers ?? new int[0];
            Machines =
                new RestrictedBoltzmannMachineD[saveInfos.Count() + (appendLayers.Length == 0 ? 0 : appendLayers.Length)
                    ];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new RestrictedBoltzmannMachineD(saveInfos[i].NumVisible, saveInfos[i].NumHidden,
                    saveInfos[i].Weights,
                    ExitConditionEvaluatorFactory.Create(i, saveInfos[i].NumVisible, saveInfos[i].NumHidden),
                    learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }

            if (appendLayers.Length > 0)
            {
                for (int j = -1; j < appendLayers.Length - 1; j++)
                {
                    Console.WriteLine("Appending Layer {0}: {1}x{2}", j + saveInfos.Count + 1,
                        j == -1 ? saveInfos.Last().NumHidden : appendLayers[j], appendLayers[j + 1]);

                    var rbm = new RestrictedBoltzmannMachineD(
                        j == -1 ? saveInfos.Last().NumHidden : appendLayers[j], appendLayers[j + 1],
                        ExitConditionEvaluatorFactory.Create(saveInfos.Count + j,
                            j == -1 ? saveInfos.Last().NumHidden : appendLayers[j], appendLayers[j + 1]), learningRate);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }
        }

        public DeepBeliefNetworkD(int[] layerSizes, double learningRate,
            IExitConditionEvaluatorFactory<double> exitConditionExitConditionEvaluatorFactory)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            Machines = new RestrictedBoltzmannMachineD[layerSizes.Length - 1];

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i], layerSizes[i + 1]);

                var rbm = new RestrictedBoltzmannMachineD(layerSizes[i], layerSizes[i + 1],
                    exitConditionExitConditionEvaluatorFactory.Create(i, layerSizes[i], layerSizes[i + 1]), learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public void TrainLayersFrom(double[,] visibleData, int startDepth)
        {
            double error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : Train(visibleData, i, out error);
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

        public double[,] Train(double[,] data, int layerNumber, out double error)
        {
            double err = Machines[layerNumber].Train(data);
            RaiseTrainEnd(err);
            error = err;
            return Machines[layerNumber].GetHiddenLayer(data);
        }

        public Task AsyncTrain(double[,] data, int layerPosition)
        {
            double err;
            return Task.Run(
                () => Train(data, layerPosition, out err));
        }

        public void TrainAll(double[,] visibleData)
        {
            double error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = Train(visibleData, i, out error);
                RaiseTrainEnd(error);
            }
        }

        public Task AsyncTrainAll(double[,] visibleData)
        {
            return Task.Run(() => TrainAll(visibleData));
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
    }
}