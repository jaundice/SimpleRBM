using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Threading;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Cuda;

namespace CudaNN
{
    public enum SuspendState
    {
        Active,
        Suspended
    }

    public interface IAdvancedRbmCuda<TElement> : IDisposable, IRestrictedBoltzmannMachine<TElement>
        where TElement : struct, IComparable<TElement>
    {
        int LayerIndex { get; }

        Matrix2D<TElement> Encode(Matrix2D<TElement> data);
        Matrix2D<TElement> Decode(Matrix2D<TElement> activations);

        SuspendState State { get; }

        void Suspend();
        void Wake();

        void GreedyTrain(Matrix2D<TElement> data,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator, CancellationToken cancelToken);

        void GreedyBatchedTrain(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator, CancellationToken cancelToken);

        void GreedyBatchedTrainMem(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluator<TElement> exitConditionEvaluator,
            ILearningRateCalculator<TElement> weightLearningRateCalculator,
            ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
            ILearningRateCalculator<TElement> visBiasLearningRateCalculator, CancellationToken cancelToken);

        void GreedyBatchedTrainMem(IList<TElement[,]> batches,
           IExitConditionEvaluator<TElement> exitConditionEvaluator,
           ILearningRateCalculator<TElement> weightLearningRateCalculator,
           ILearningRateCalculator<TElement> hidBiasLearningRateCalculator,
           ILearningRateCalculator<TElement> visBiasLearningRateCalculator, CancellationToken cancelToken);

        Matrix2D<TElement> HiddenBiases { get; }
        Matrix2D<TElement> VisibleBiases { get; }
        Matrix2D<TElement> Weights { get; }
        TElement FinalMomentum { get; }
        TElement InitialMomentum { get; }
        TElement WeightCost { get; }
        GPGPU GPU { get; }
        GPGPURAND GPURAND { get; }
        int NumVisibleNeurons { get; }
        int NumHiddenNeurons { get; }
        Matrix2D<TElement> Reconstruct(Matrix2D<TElement> data);
        void Dispose(bool disposing);

        void SetState(SuspendState state);

        void Save(string path);
    }
}