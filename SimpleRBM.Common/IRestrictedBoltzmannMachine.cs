using System;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IRestrictedBoltzmannMachine<T> where T : struct, IComparable<T>
    {
        int NumHiddenElements { get; }
        int NumVisibleElements { get; }
        ILearningRateCalculator<T> LearningRate { get; }
        IExitConditionEvaluator<T> ExitConditionEvaluator { get; }
        T[,] GetHiddenLayer(T[,] visibleStates);
        T[,] GetSoftmaxLayer(T[,] visibleStates);

        T[,] GetVisibleLayer(T[,] hiddenStates);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfSamples);

        T GreedyTrain(T[][] data);
        Task<T> AsyncGreedyTrain(T[][] data);
        T GreedyTrain(T[,] visibleData);

        T GreedySupervisedTrain(T[,] data, T[,] labels);
        T GreedyBatchedSupervisedTrain(T[,] data, T[,] labels, int batchSize);

        T[,] Classify(T[,] data, out T[,] labels);

        Task<T> AsyncGreedyTrain(T[,] data);

        T GreedyBatchedTrain(T[][] data, int batchRows);
        Task<T> AsyncGreedyBatchedTrain(T[][] data, int batchRows);
        T GreedyBatchedTrain(T[,] data, int batchRows);
        Task<T> AsyncGreedyBatchedTrain(T[,] data, int batchRows);

        ILayerSaveInfo<T> GetSaveInfo();

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;

        T CalculateReconstructionError(T[,] data);
    }
}