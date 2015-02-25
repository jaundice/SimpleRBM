using System;
using System.Collections.Generic;
using SimpleRBM.Common;
using SimpleRBM.Cuda;

namespace CudaNN
{
    public interface ICudaNetwork<TElement> : IDisposable, INetwork<TElement>
        where TElement : struct, IComparable<TElement>
    {
        void SetDefaultMachineState(SuspendState state);

        new IList<IAdvancedRbmCuda<TElement>> Machines { get; }
        Matrix2D<TElement> Encode(Matrix2D<TElement> data, int maxDepth = -1);

        /// <summary>
        /// Same as Encode except the visible data is extended for the inner most rbm to the size of the visible buffer (i.e numlabels columns are added)
        /// </summary>
        /// <param name="data"></param>
        /// <param name="maxDepth"></param>
        /// <returns></returns>
        Matrix2D<TElement> EncodeWithLabelExpansion(Matrix2D<TElement> data);

        Matrix2D<TElement> Reconstruct(Matrix2D<TElement> data, int maxDepth = -1);
        Matrix2D<TElement> Decode(Matrix2D<TElement> activations, int maxDepth = -1);

        Matrix2D<TElement> ReconstructWithLabels(Matrix2D<TElement> data, out Matrix2D<TElement> labels,
            bool softmaxLabels = true);

        Matrix2D<TElement> DecodeWithLabels(Matrix2D<TElement> activations,
            out Matrix2D<TElement> labels, bool softmaxLabels = true);

        Matrix2D<TElement> LabelData(Matrix2D<TElement> data, bool softmaxLabels = true);
        new Matrix2D<TElement> Daydream(int numDreams, int maxDepth = -1, bool guassian = true);

        Matrix2D<TElement> DaydreamWithLabels(int numDreams, out Matrix2D<TElement> labels, bool guassian = true,
            bool softmaxLabels = true);

        Matrix2D<TElement> DaydreamByClass(Matrix2D<TElement> modelLabels,
            out Matrix2D<TElement> generatedLabels, bool guassian = true, bool softmaxLabels = true);

        void GreedyTrain(Matrix2D<TElement> data,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

        void GreedyBatchedTrain(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

        void GreedyBatchedTrainMem(Matrix2D<TElement> data, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

        void GreedySupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

        void GreedyBatchedSupervisedTrain(Matrix2D<TElement> data, Matrix2D<TElement> labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

        void GreedyBatchedSupervisedTrainMem(Matrix2D<TElement> data, Matrix2D<TElement> labels, int batchSize,
            IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
            ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
            ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

     //   void GreedyBatchedTrain(IList<Matrix2D<TElement>> batches, 
     //      IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
     //      ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
     //      ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
     //      ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

     //   void GreedyBatchedTrainMem(IList<Matrix2D<TElement>> batches,
     //       IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
     //       ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
     //       ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
     //       ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

     //   void GreedyBatchedSupervisedTrain(IList<Matrix2D<TElement>> batches, IList<Matrix2D<TElement>> labels, int batchSize,
     //       IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
     //       ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
     //       ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
     //       ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);

     //void GreedyBatchedSupervisedTrainMem(IList<Matrix2D<TElement>> batches, IList<Matrix2D<TElement>> labels, int batchSize,
     //       IExitConditionEvaluatorFactory<TElement> exitConditionFactory,
     //       ILearningRateCalculatorFactory<TElement> weightLearningRateCalculatorFactory,
     //       ILearningRateCalculatorFactory<TElement> hidBiasLearningRateCalculatorFactory,
     //       ILearningRateCalculatorFactory<TElement> visBiasLearningRateCalculatorFactory);
    }
}