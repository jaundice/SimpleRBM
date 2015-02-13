using System;
using System.Collections.Generic;
using SimpleRBM.Common;

namespace CudaNN
{
    public interface INetwork<TElementType> where TElementType : struct, IComparable<TElementType>
    {
        IList<IRbm<TElementType>> Machines { get; }

        event EventHandler<EpochEventArgs<TElementType>> EpochComplete;
        event EventHandler<EpochEventArgs<TElementType>> LayerTrainComplete;
        TElementType[,] Reconstruct(TElementType[,] data, int maxDepth = -1);
        TElementType[,] Encode(TElementType[,] data, int maxDepth = -1);
        TElementType[,] Decode(TElementType[,] activations, int maxDepth = -1);

        TElementType[,] ReconstructWithLabels(TElementType[,] data,
            out TElementType[,] labels, bool softmaxLabels = true);

        TElementType[,] DecodeWithLabels(TElementType[,] activations,
            out TElementType[,] labels, bool softmaxLabels = true);

        TElementType[,] LabelData(TElementType[,] data, bool softmaxLabels = true);

        TElementType[,] Daydream(int numDreams, int maxDepth = -1, bool guassian = true);

        TElementType[,] DaydreamWithLabels(int numDreams, out TElementType[,] labels,
            bool guassian = true, bool softmaxLabels = true);

        TElementType[,] DaydreamByClass(TElementType[,] modelLabels,
            out TElementType[,] generatedLabels, bool guassian = true);

        void GreedyTrain(TElementType[,] data,
            IExitConditionEvaluatorFactory<TElementType> exitConditionFactory,
            ILearningRateCalculatorFactory<TElementType> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> visBiasLearningRateCalculatorFactory);

        void GreedySupervisedTrain(TElementType[,] data, TElementType[,] labels,
            IExitConditionEvaluatorFactory<TElementType> exitConditionFactory,
            ILearningRateCalculatorFactory<TElementType> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElementType> visBiasLearningRateCalculatorFactory);
    }
}