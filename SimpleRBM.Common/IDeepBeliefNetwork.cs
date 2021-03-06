﻿using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IDeepBeliefNetworkExtended<TElement> : IDeepBeliefNetwork<TElement> where TElement : struct, IComparable<TElement>
    {
        TElement[,] Encode(TElement[,] data, int maxDepth);
        TElement[,] Decode(TElement[,] data, int maxDepth);
        TElement[,] Reconstruct(TElement[,] data, int maxDepth);
        TElement[,] DayDream(int numberOfDreams, int maxDepth);
        TElement[,] DaydreamByClass(TElement[,] labels);

        void GreedySupervisedTrainAll(TElement[,] srcData, TElement[,] labels, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory, CancellationToken cancelToken);
        void GreedyBatchedSupervisedTrainAll(TElement[,] srcData, TElement[,] labels, int batchSize, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory, CancellationToken cancelToken);

        TElement[,] ReconstructWithLabels(TElement[,] data, out TElement[,] labels);

        void UpDownTrainAll(TElement[,] visibleData, int iterations, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory, CancellationToken cancelToken);

        void UpDownTrainSupervisedAll(TElement[,] visibleData, TElement[,] labels, int iterations, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory, CancellationToken cancelToken);
    }

    public interface IDeepBeliefNetwork<TElement> where TElement : struct, IComparable<TElement>
    {
        int NumMachines { get; }
        TElement[,] Encode(TElement[,] data);
        TElement[,] Decode(TElement[,] data);
        TElement[,] Reconstruct(TElement[,] data);
        TElement[,] DayDream(int numberOfDreams);
        void GreedyTrainAll(TElement[,] visibleData, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory, CancellationToken cancelToken);
        void GreedyBatchedTrainAll(TElement[,] visibleData, int batchRows, IExitConditionEvaluatorFactory<TElement> exitConditionEvaluatorFactory, ILearningRateCalculatorFactory<TElement> learningRateFactory, CancellationToken cancelToken);
        IEnumerable<ILayerSaveInfo<TElement>> GetLayerSaveInfos();

        event EventHandler<EpochEventArgs<TElement>> EpochEnd;
        event EventHandler<EpochEventArgs<TElement>> TrainEnd;
    }
}