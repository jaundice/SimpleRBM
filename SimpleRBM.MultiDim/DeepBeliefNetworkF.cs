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
    public class DeepBeliefNetworkF : IDeepBeliefNetwork<float>
    {
        private readonly RestrictedBoltzmannMachineF[] Machines;

        public DeepBeliefNetworkF(DirectoryInfo network, float learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory, int[] appendLayers = null)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            List<LayerSaveInfoF> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoF(a.FullName)).ToList();

            appendLayers = appendLayers ?? new int[0];
            Machines =
                new RestrictedBoltzmannMachineF[saveInfos.Count() + (appendLayers.Length == 0 ? 0 : appendLayers.Length)
                    ];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new RestrictedBoltzmannMachineF(saveInfos[i].NumVisible, saveInfos[i].NumHidden,
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

                    var rbm = new RestrictedBoltzmannMachineF(
                        j == -1 ? saveInfos.Last().NumHidden : appendLayers[j], appendLayers[j + 1],
                        ExitConditionEvaluatorFactory.Create(saveInfos.Count + j,
                            j == -1 ? saveInfos.Last().NumHidden : appendLayers[j], appendLayers[j + 1]), learningRate);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }
        }

        public DeepBeliefNetworkF(int[] layerSizes, float learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory)
        {
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            Machines = new RestrictedBoltzmannMachineF[layerSizes.Length - 1];

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i], layerSizes[i + 1]);

                var rbm = new RestrictedBoltzmannMachineF(layerSizes[i], layerSizes[i + 1],
                    exitConditionExitConditionEvaluatorFactory.Create(i, layerSizes[i], layerSizes[i + 1]), learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public void TrainLayersFrom(float[,] visibleData, int startDepth)
        {
            float error;
            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = i < startDepth
                    ? Machines[i].GetHiddenLayer(visibleData)
                    : Train(visibleData, i, out error);
            }
        }

        public float[,] Encode(float[,] data)
        {
            data = Machines[0].GetHiddenLayer(data);

            for (int i = 0; i < Machines.Length - 1; i++)
            {
                data = Machines[i + 1].GetHiddenLayer(data);
            }

            return data;
        }

        public float[,] Decode(float[,] data)
        {
            data = Machines[Machines.Length - 1].GetVisibleLayer(data);

            for (int i = Machines.Length - 1; i > 0; i--)
            {
                data = Machines[i - 1].GetVisibleLayer(data);
            }

            return data;
        }

        public float[,] Reconstruct(float[,] data)
        {
            float[,] hl = Encode(data);
            return Decode(hl);
        }

        public float[,] DayDream(int numberOfDreams)
        {
            float[,] dreamRawData = Distributions.UniformRandromMatrixBoolF(numberOfDreams,
                Machines[0].NumVisibleElements);

            float[,] ret = Reconstruct(dreamRawData);

            return ret;
        }

        public float[,] Train(float[,] data, int layerNumber, out float error)
        {
            float err = Machines[layerNumber].Train(data);
            RaiseTrainEnd(err);
            error = err;
            return Machines[layerNumber].GetHiddenLayer(data);
        }

        public Task AsyncTrain(float[,] data, int layerPosition)
        {
            float err;
            return Task.Run(
                () => Train(data, layerPosition, out err));
        }

        public void TrainAll(float[,] visibleData)
        {
            float error;

            for (int i = 0; i < Machines.Length; i++)
            {
                visibleData = Train(visibleData, i, out error);
                RaiseTrainEnd(error);
            }
        }

        public Task AsyncTrainAll(float[,] visibleData)
        {
            return Task.Run(() => TrainAll(visibleData));
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
    }
}