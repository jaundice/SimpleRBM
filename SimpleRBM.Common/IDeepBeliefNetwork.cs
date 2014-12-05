using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IDeepBeliefNetwork<T> where T : struct, IComparable<T>
    {
        int NumMachines { get; }
        IExitConditionEvaluatorFactory<T> ExitConditionEvaluatorFactory { get; }

        T[,] Encode(T[,] data);
        T[,] Decode(T[,] data);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfDreams);


        T[,] Train(T[,] data, int layerPosition, out T error);
        Task AsyncTrain(T[,] data, int layerPosition);
        void TrainAll(T[,] visibleData);
        Task AsyncTrainAll(T[,] visibleData);

        void TrainLayersFrom(T[,] visibleData, int startDepth);
        

        IEnumerable<ILayerSaveInfo<T>> GetLayerSaveInfos();

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;
    }

    public class EpochEventArgs<T> : EventArgs
    {
        public int Epoch { get; set; }
        public T Error { get; set; }
    }
}