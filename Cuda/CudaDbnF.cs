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
    public class CudaDbnF : IDeepBeliefNetwork<float>, IDisposable
    {
        private readonly CudaRbmF[] Machines;

        public int NumMachines
        {
            get { return Machines.Length; }
        }
        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;


        public CudaDbnF(GPGPU gpu, GPGPURAND rand, DirectoryInfo network, float learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory, int[] appendLayers = null)
        {
            _gpu = gpu;
            _rand = rand;
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;
            List<LayerSaveInfoF> saveInfos =
                network.GetFiles("*.bin")
                    .OrderBy(a => int.Parse(Regex.Match(Path.GetFileNameWithoutExtension(a.Name), "[0-9]+").Value))
                    .Select(a => new LayerSaveInfoF(a.FullName)).ToList();

            appendLayers = appendLayers ?? new int[0];
            Machines =
                new CudaRbmF[saveInfos.Count() + (appendLayers.Length == 0 ? 0 : appendLayers.Length)
                    ];


            for (int i = 0; i < saveInfos.Count; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, saveInfos[i].NumVisible, saveInfos[i].NumHidden);

                var rbm = new CudaRbmF(gpu, rand, saveInfos[i].NumVisible, saveInfos[i].NumHidden,
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

                    var rbm = new CudaRbmF(gpu, rand,
                        j == -1 ? saveInfos.Last().NumHidden : appendLayers[j], appendLayers[j + 1],
                        ExitConditionEvaluatorFactory.Create(saveInfos.Count + j,
                            j == -1 ? saveInfos.Last().NumHidden : appendLayers[j], appendLayers[j + 1]), learningRate);
                    rbm.EpochEnd += OnRbm_EpochEnd;
                    Machines[saveInfos.Count + j + 1] = rbm;
                }
            }
        }

        public CudaDbnF(GPGPU gpu, GPGPURAND rand, int[] layerSizes, float learningRate,
            IExitConditionEvaluatorFactory<float> exitConditionExitConditionEvaluatorFactory)
        {
            _gpu = gpu;
            _rand = rand;
            ExitConditionEvaluatorFactory = exitConditionExitConditionEvaluatorFactory;


            Machines = new CudaRbmF[layerSizes.Length - 1];

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                Console.WriteLine("Building Layer {0}: {1}x{2}", i, layerSizes[i], layerSizes[i + 1]);

                var rbm = new CudaRbmF(gpu, rand, layerSizes[i], layerSizes[i + 1],
                    ExitConditionEvaluatorFactory.Create(i, layerSizes[i], layerSizes[i + 1]), learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                Machines[i] = rbm;
            }
        }

        public bool Disposed { get; protected set; }

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
            int elems = Machines[0].NumVisibleElements;
            using (
                Matrix2D<float> dreamRawData = CudaRbmF.UniformDistribution(_gpu, _rand,
                    numberOfDreams, elems))
            {
                dim3 grid, block;
                ThreadOptimiser.Instance.GetStrategy(numberOfDreams, elems, out grid, out block);

                _gpu.Launch(grid, block, Matrix2DCuda.ToBinary, dreamRawData.Matrix);

                var localRaw = new float[numberOfDreams, elems];
                _gpu.CopyFromDevice(dreamRawData, localRaw);
                float[,] ret = Reconstruct(localRaw);
                return ret;
            }
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
    }
}