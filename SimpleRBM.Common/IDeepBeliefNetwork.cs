using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IDeepBeliefNetworkExtended<T> : IDeepBeliefNetwork<T> where T : struct, IComparable<T>
    {
        T[,] Classify(T[,] data, int maxDepth);
        T[,] Encode(T[,] data, int maxDepth);
        T[,] Decode(T[,] data, int maxDepth);
        T[,] Reconstruct(T[,] data, int maxDepth);
        T[,] DayDream(int numberOfDreams, int maxDepth);
        T[,] GenerateExamplesByLabel(T[,] labels);
        T GetReconstructionError(T[,] srcData, int depth);
        T GreedySupervisedTrainAll(T[,] srcData, T[,] labels, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);
        T GreedyBatchedSupervisedTrainAll(T[,] srcData, T[,] labels, int batchSize, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);

        T[,] GreedySupervisedTrain(T[,] data, T[,] labels, int layerPosition, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory, out T error, out T[,] labelsPredicted);

        T[,] Classify(T[,] data, out T[,] labels);

        void UpDownTrainAll(T[,] visibleData, int iterations, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);

        void UpDownTrainSupervisedAll(T[,] visibleData, T[,] labels, int iterations, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);
    }

    public interface IDeepBeliefNetwork<T> where T : struct, IComparable<T>
    {
        int NumMachines { get; }
        //IExitConditionEvaluatorFactory<T> ExitConditionEvaluatorFactory { get; }

        T[,] Encode(T[,] data);
        T[,] Decode(T[,] data);
        T[,] Reconstruct(T[,] data);


        T[,] DayDream(int numberOfDreams);


        T[,] GreedyTrain(T[,] data, int layerIndex, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory, out T error);
        Task AsyncGreedyTrain(T[,] data, int layerIndex, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);
        void GreedyTrainAll(T[,] visibleData, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);
        Task AsyncGreedyTrainAll(T[,] visibleData, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);

        void GreedyTrainLayersFrom(T[,] visibleData, int startDepth, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);


        T[,] GreedyBatchedTrain(T[,] data, int layerPosition, int batchRows, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory, out T error);
        Task AsyncGreedyBatchedTrain(T[,] data, int layerPosition, int batchRows, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);
        void GreedyBatchedTrainAll(T[,] visibleData, int batchRows, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);
        Task AsyncGreedyBatchedTrainAll(T[,] visibleData, int batchRows, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);

        void GreedyBatchedTrainLayersFrom(T[,] visibleData, int startDepth, int batchRows, IExitConditionEvaluatorFactory<T> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<T> learningRateFactory);


        IEnumerable<ILayerSaveInfo<T>> GetLayerSaveInfos();

        event EventHandler<EpochEventArgs<T>> EpochEnd;
        event EventHandler<EpochEventArgs<T>> TrainEnd;
    }

    public class EpochEventArgs<T> : EventArgs
    {
        public int Layer { get; set; }
        public int Epoch { get; set; }
        public T Error { get; set; }
    }
}