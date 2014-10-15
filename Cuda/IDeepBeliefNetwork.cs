using System;
using System.Threading.Tasks;

namespace CudaRbm
{
    public interface IDeepBeliefNetwork<T>
    {
        T[,] Encode(T[,] data);
        T[,] Decode(T[,] data);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfDreams);


        T[,] Train(T[,] data, int layerPosition, out T error);
        Task AsyncTrain(T[,] data, int layerPosition);
        void TrainAll(T[,] visibleData);
        Task AsyncTrainAll(T[,] visibleData);

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;
    }

    public class EpochEventArgs<T> : EventArgs
    {
        public int Epoch { get; set; }
        public T Error { get; set; }
    }
}