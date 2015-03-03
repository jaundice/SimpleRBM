using System;
using System.Threading;
using System.Threading.Tasks;

namespace SimpleRBM.Common
{
    public interface IRestrictedBoltzmannMachine<TElement> where TElement : struct, IComparable<TElement>
    {
        int NumHiddenNeurons { get; }
        int NumVisibleNeurons { get; }
     
        TElement[,] Encode(TElement[,] visibleStates);

        TElement[,] Decode(TElement[,] hiddenStates);
        TElement[,] Reconstruct(TElement[,] data);
        TElement[,] DayDream(int numberOfSamples);

        void GreedyTrain(TElement[,] visibleData, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken);

        ILayerSaveInfo<TElement> GetSaveInfo();

        event EventHandler<EpochEventArgs<TElement>> EpochEnd;
        event EventHandler<EpochEventArgs<TElement>> TrainEnd;

        TElement CalculateReconstructionError(TElement[,] data);
    }
}