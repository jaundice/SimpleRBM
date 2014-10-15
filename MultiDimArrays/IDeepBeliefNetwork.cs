using System;
using System.Threading.Tasks;

namespace MultidimRBM
{
    public interface IDeepBeliefNetwork<T>
    {
        T[,] Encode(T[,] data);
        T[,] Decode(T[,] data);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfDreams);


        T[,] Train(T[,] data, int layerPosition, out double error);
        Task AsyncTrain(T[,] data, int layerPosition);
        void TrainAll(T[,] visibleData);
        Task AsyncTrainAll(T[,] visibleData);

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;

    }
}