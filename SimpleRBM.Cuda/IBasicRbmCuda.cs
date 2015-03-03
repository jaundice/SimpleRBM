using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public interface IBasicRbmCuda<TElement> : IRestrictedBoltzmannMachine<TElement> , IDisposable where TElement : struct, IComparable<TElement>
    {
        GPGPU GPU { get; }
        GPGPURAND GPURAND { get; }
        Matrix2D<TElement> Encode(Matrix2D<TElement> visibleStates);
        Matrix2D<TElement> Decode(Matrix2D<TElement> hiddenStates);
        Matrix2D<TElement> Reconstruct(Matrix2D<TElement> data);
        new Matrix2D<TElement> DayDream(int numberOfSamples);

        TElement GreedyTrain(Matrix2D<TElement> visibleData, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken);


        TElement GreedyBatchedTrain(Matrix2D<TElement> data, int batchRows,
            IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken);

        TElement CalculateReconstructionError(Matrix2D<TElement> data);

        void DownPass(Matrix2D<TElement> hiddenStates, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, out TElement error, CancellationToken cancelToken);
    }
}