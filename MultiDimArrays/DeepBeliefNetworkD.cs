using System;
using System.Threading.Tasks;

namespace MultidimRBM
{
    public class DeepBeliefNetworkD : IDeepBeliefNetwork<double>
    {
        private readonly RestrictedBoltzmannMachineD[] Machines;

        public DeepBeliefNetworkD(int[] layerSizes, double learningRate)
        {
            Machines = new RestrictedBoltzmannMachineD[layerSizes.Length - 1];

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                var rbm = new RestrictedBoltzmannMachineD(layerSizes[i], layerSizes[i + 1], learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
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
            double[,] dreamRawData = Distributions.UniformRandromMatrixBool(numberOfDreams,
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