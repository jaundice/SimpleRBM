using System;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IRestrictedBoltzmannMachine<T> where T : struct, IComparable<T>
    {
        int NumHiddenNeurons { get; }
        int NumVisibleNeurons { get; }
     
        T[,] Encode(T[,] visibleStates);

        T[,] Decode(T[,] hiddenStates);
        T[,] Reconstruct(T[,] data);
        T[,] DayDream(int numberOfSamples);

        T GreedyTrain(T[,] visibleData, IExitConditionEvaluator<T> exitEvaluator,
            ILearningRateCalculator<T> learningRateCalculator);

        //T GreedySupervisedTrain(T[,] data, T[,] labels, IExitConditionEvaluator<T> exitEvaluator,
        //    ILearningRateCalculator<T> learningRateCalculator);

        //T GreedyBatchedSupervisedTrain(T[,] data, T[,] labels, int batchSize, IExitConditionEvaluator<T> exitEvaluator,
        //    ILearningRateCalculator<T> learningRateCalculator);

        //T GreedyBatchedTrain(T[,] data, int batchRows, IExitConditionEvaluator<T> exitEvaluator,
        //    ILearningRateCalculator<T> learningRateCalculator);


        ILayerSaveInfo<T> GetSaveInfo();

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;

        T CalculateReconstructionError(T[,] data);
    }
}