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
        T GetReconstructionError(T[,] srcData, int depth);
        T GreedySupervisedTrainAll(T[,] srcData, T[,] labels);
        T GreedyBatchedSupervisedTrainAll(T[,] srcData, T[,] labels, int batchSize);

        T[,] GreedySupervisedTrain(T[,] data, T[,] labels, int layerPosition, out T error, out T[,] labelsPredicted);

        T[,] Classify(T[,] data, out T[,] labels);

        void UpDownTrainAll(T[,] visibleData, int iterations, int epochsPerMachine, T learningRate);

        void UpDownTrainSupervisedAll(T[,] visibleData, T[,] labels, int iterations, int epochsPerMachine,
            T learningRate);
    }

    public interface IDeepBeliefNetwork<T> where T : struct, IComparable<T>
    {
        int NumMachines { get; }
        IExitConditionEvaluatorFactory<T> ExitConditionEvaluatorFactory { get; }

        T[,] Encode(T[,] data);
        T[,] Decode(T[,] data);
        T[,] Reconstruct(T[,] data);


        T[,] DayDream(int numberOfDreams);


        T[,] GreedyTrain(T[,] data, int layerPosition, out T error);
        Task AsyncGreedyTrain(T[,] data, int layerPosition);
        void GreedyTrainAll(T[,] visibleData);
        Task AsyncGreedyTrainAll(T[,] visibleData);

        void GreedyTrainLayersFrom(T[,] visibleData, int startDepth);


        T[,] GreedyBatchedTrain(T[,] data, int layerPosition, int batchRows, out T error);
        Task AsyncGreedyBatchedTrain(T[,] data, int layerPosition, int batchRows);
        void GreedyBatchedTrainAll(T[,] visibleData, int batchRows);
        Task AsyncGreedyBatchedTrainAll(T[,] visibleData, int batchRows);

        void GreedyBatchedTrainLayersFrom(T[,] visibleData, int startDepth, int batchRows);


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