using System;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;

namespace CudaRbm
{
    public class DeepBeliefNetworkF : IDeepBeliefNetwork<float>
    {
        private readonly RestrictedBoltzmannMachineF[] Machines;
        private GPGPU _gpu;
        private GPGPURAND _rand;
        private dim3 _block;
        private dim3 _grid;

        public DeepBeliefNetworkF(GPGPU gpu, GPGPURAND rand, dim3 grid, dim3 block, int[] layerSizes, float learningRate)
        {
            _gpu = gpu;
            _rand = rand;
            _grid = grid;
            _block = block;

            Machines = new RestrictedBoltzmannMachineF[layerSizes.Length - 1];

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                Console.WriteLine("Building Layer {0}", i);

                var rbm = new RestrictedBoltzmannMachineF(gpu, rand, grid, block, layerSizes[i], layerSizes[i + 1], learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
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

            var elems = Machines[0].NumVisibleElements;
            var dreamRawData = _gpu.Allocate<float>(numberOfDreams, elems);
            RestrictedBoltzmannMachineF.UniformDistribution(_gpu, _rand, dreamRawData, numberOfDreams, elems);
            _gpu.Launch(_grid, _block, Matrix2D.ToBinary, dreamRawData);

            var localRaw = new float[numberOfDreams, elems];
            _gpu.CopyFromDevice(dreamRawData, localRaw);
            _gpu.Free(dreamRawData);
            
            
            float[,] ret = Reconstruct(localRaw);



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