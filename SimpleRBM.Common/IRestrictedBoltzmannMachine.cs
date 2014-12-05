using System;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IRestrictedBoltzmannMachine<T> where T : struct, IComparable<T>
    {
        int NumHiddenElements { get; }
        int NumVisibleElements { get; }
        T LearningRate { get; }
        IExitConditionEvaluator<T> ExitConditionEvaluator { get; }
        T[,] GetHiddenLayer(T[,] data);
        T[,] GetVisibleLayer(T[,] data);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfSamples);

        T Train(T[][] data);
        Task<T> AsyncTrain(T[][] data);
        T Train(T[,] data);
        Task<T> AsyncTrain(T[,] data);

        ILayerSaveInfo<T> GetSaveInfo(); 

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;
    }
}