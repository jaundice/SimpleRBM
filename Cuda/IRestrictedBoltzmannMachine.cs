using System;
using System.Threading.Tasks;
using Cudafy.Host;

namespace CudaRbm
{
    public interface IRestrictedBoltzmannMachine<T>
    {
        T[,] GetHiddenLayer(T[,] data);
        T[,] GetVisibleLayer(T[,] data);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfSamples);


        T Train(T[][] data);
        Task<T> AsyncTrain(T[][] data);
        T Train(T[,] data);
        Task<T> AsyncTrain(T[,] data);

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;
    }
}