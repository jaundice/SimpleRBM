using System;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Cuda;

namespace CudaNN
{
    public interface IAdvancedRbmCuda<TElementType> : IDisposable, IRestrictedBoltzmannMachine<TElementType>
        where TElementType : struct, IComparable<TElementType>
    {
        Matrix2D<TElementType> Encode(Matrix2D<TElementType> data);
        Matrix2D<TElementType> Decode(Matrix2D<TElementType> activations);

        void GreedyTrain(Matrix2D<TElementType> data,
            IExitConditionEvaluator<TElementType> exitConditionEvaluator,
            ILearningRateCalculator<TElementType> weightLearningRateCalculator,
            ILearningRateCalculator<TElementType> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElementType> visBiasLearningRateCalculator);

        Matrix2D<TElementType> HiddenBiases { get; }
        Matrix2D<TElementType> VisibleBiases { get; }
        Matrix2D<TElementType> Weights { get; }
        TElementType FinalMomentum { get; }
        TElementType InitialMomentum { get; }
        TElementType WeightCost { get; }
        GPGPU GPU { get; }
        GPGPURAND GPURAND { get; }
        int NumVisibleNeurons { get; }
        int NumHiddenNeurons { get; }
        Matrix2D<TElementType> Reconstruct(Matrix2D<TElementType> data);
        void Dispose(bool disposing);
    }
}