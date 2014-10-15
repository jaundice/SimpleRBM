using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultidimRBM
{
    public interface IRestrictedBoltzmannMachine<T>
    {

        T[,] GetHiddenLayer(T[,] data);
        T[,] GetVisibleLayer(T[,] data);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfSamples);


        T Train(double[][] data);
        Task<T> AsyncTrain(double[][] data);
        T Train(double[,] data);
        Task<T> AsyncTrain(double[,] data);

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;

    }

    public class EpochEventArgs<T> : EventArgs
    {
        public int Epoch { get; set; }
        public T Error { get; set; }
    }
}
