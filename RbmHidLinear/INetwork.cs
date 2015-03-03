using System;
using System.Collections.Generic;
using System.Threading;
using SimpleRBM.Common;

namespace CudaNN
{
    public interface INetwork<TElement> where TElement : struct, IComparable<TElement>
    {
        IList<IRestrictedBoltzmannMachine<TElement>> Machines { get; }

        event EventHandler<EpochEventArgs<TElement>> EpochComplete;
        event EventHandler<EpochEventArgs<TElement>> LayerTrainComplete;
        TElement[,] Reconstruct(TElement[,] data, int maxDepth = -1);
        TElement[,] Encode(TElement[,] data, int maxDepth = -1);
        TElement[,] Decode(TElement[,] activations, int maxDepth = -1);

        TElement[,] ReconstructWithLabels(TElement[,] data,
            out TElement[,] labels, bool softmaxLabels = true);

        TElement[,] DecodeWithLabels(TElement[,] activations,
            out TElement[,] labels, bool softmaxLabels = true);

        TElement[,] LabelData(TElement[,] data, bool softmaxLabels = true);

        TElement[,] Daydream(int numDreams, int maxDepth = -1, bool guassian = true);

        TElement[,] DaydreamWithLabels(int numDreams, out TElement[,] labels,
            bool guassian = true, bool softmaxLabels = true);

        TElement[,] DaydreamByClass(TElement[,] modelLabels,
            out TElement[,] generatedLabels, bool guassian = true, bool softmaxGeneratedLabels = true);

        void GreedyTrain(TElement[,] data,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);

        void GreedyBatchedTrain(TElement[,] data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);

        void GreedyBatchedTrainMem(TElement[,] data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);

        void GreedySupervisedTrain(TElement[,] data, TElement[,] labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);

        void GreedyBatchedSupervisedTrain(TElement[,] data, TElement[,] labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);

        void GreedyBatchedSupervisedTrainMem(TElement[,] data, TElement[,] labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);

     void GreedyBatchedTrainMem(IList<TElement[,]> batches,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);

     void GreedyBatchedSupervisedTrainMem(IList<TElement[,]> batches, IList<TElement[,]> labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory, CancellationToken cancelToken);
    }
}