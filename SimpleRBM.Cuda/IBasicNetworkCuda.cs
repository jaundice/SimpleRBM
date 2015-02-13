using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public interface IBasicNetworkCuda<TElement> : IDisposable, IDeepBeliefNetworkExtended<TElement>
        where TElement : struct, IComparable<TElement>
    {
        IList<IBasicRbmCuda<TElement>> Machines { get; }

        Matrix2D<TElement> Encode(Matrix2D<TElement> data, int maxDepth = -1);
        Matrix2D<TElement> Decode(Matrix2D<TElement> data, int maxDepth = -1);
        Matrix2D<TElement> Reconstruct(Matrix2D<TElement> data, int maxDepth = -1);
        Matrix2D<TElement> DayDream(int numberOfDreams, int maxDepth = -1, bool guassian = true);

        Matrix2D<TElement> DaydreamByClass(Matrix2D<TElement> modelLabels,
            out Matrix2D<TElement> generatedLabels, bool guassian = true, bool softmaxLabels = true);

        void GreedyTrainAll(Matrix2D<TElement> visibleData,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory);

        void GreedyTrainLayersFrom(Matrix2D<TElement> visibleData, int startDepth,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory);

        void GreedyBatchedTrainAll(Matrix2D<TElement> data, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory, out TElement error);

        void UpDownTrainAll(Matrix2D<TElement> visibleData, int iterations,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory);

        Matrix2D<TElement> DecodeWithLabels(Matrix2D<TElement> activations,
            out Matrix2D<TElement> labels, bool softmaxLabels = true);

        Matrix2D<TElement> ReconstructWithLabels(Matrix2D<TElement> data, out Matrix2D<TElement> labels,
            bool softmaxLabels = true);

        Matrix2D<TElement> EncodeWithLabelExpansion(Matrix2D<TElement> data);

        void GreedySupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory);

        void GreedyBatchedSupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels, int batchRows,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory);

        void UpDownSupervisedTrainAll(Matrix2D<TElement> visibleData, Matrix2D<TElement> labels, int iterations,
            IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory,
            ILearningRateCalculatorFactory<TElement> learningRateFactory);
    }
}