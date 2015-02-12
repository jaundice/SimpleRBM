using System;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IRestrictedBoltzmannMachine<T> where T : struct, IComparable<T>
    {
        int NumHiddenElements { get; }
        int NumVisibleElements { get; }
        //ILearningRateCalculator<T> LearningRate { get; }
        //IExitConditionEvaluator<T> ExitConditionEvaluator { get; }
        T[,] GetHiddenLayer(T[,] visibleStates);
        //T[,] GetSoftmaxLayer(T[,] visibleStates);

        T[,] GetVisibleLayer(T[,] hiddenStates);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfSamples);

        ActivationFunction VisibleActivation { get; }
        ActivationFunction HiddenActivation { get; }

        T GreedyTrain(T[][] data, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        Task<T> AsyncGreedyTrain(T[][] data, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        T GreedyTrain(T[,] visibleData, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        T GreedySupervisedTrain(T[,] data, T[,] labels, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        T GreedyBatchedSupervisedTrain(T[,] data, T[,] labels, int batchSize, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        T[,] Classify(T[,] data, out T[,] labels);

        Task<T> AsyncGreedyTrain(T[,] data, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        T GreedyBatchedTrain(T[][] data, int batchRows, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        Task<T> AsyncGreedyBatchedTrain(T[][] data, int batchRows, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        T GreedyBatchedTrain(T[,] data, int batchRows, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        Task<T> AsyncGreedyBatchedTrain(T[,] data, int batchRows, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        ILayerSaveInfo<T> GetSaveInfo();

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;

        T CalculateReconstructionError(T[,] data);
    }
}