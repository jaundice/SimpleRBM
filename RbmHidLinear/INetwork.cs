using System;
using System.Collections.Generic;
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
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

        void GreedySupervisedTrain(TElement[,] data, TElement[,] labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);
    }
}