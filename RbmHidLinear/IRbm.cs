using System;
using SimpleRBM.Common;

namespace CudaNN
{
    public interface IRbm<TElementType> where TElementType : struct, IComparable<TElementType>
    {
        event EventHandler<EpochEventArgs<TElementType>> EpochEnd;
        event EventHandler<EpochEventArgs<TElementType>> TrainEnd;
        int LayerIndex { get; }
        int NumVisibleNeurons { get; }
        int NumHiddenNeurons { get; }
        void GreedyTrain(TElementType[,] visibleData, IExitConditionEvaluator<TElementType> exitConditionEvaluator, ILearningRateCalculator<TElementType> weightLearningRateCalculator, ILearningRateCalculator<TElementType> hidBiasLearningRateCalculator, ILearningRateCalculator<TElementType> visBiasLearningRateCalculator);
        TElementType[,] Encode(TElementType[,] srcData);
        TElementType[,] Decode(TElementType[,] activations);
        TElementType[,] Reconstruct(TElementType[,] data);
    }
}