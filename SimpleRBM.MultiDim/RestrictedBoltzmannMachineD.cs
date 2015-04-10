using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;

using TElement = System.Double;

namespace SimpleRBM.MultiDim
{
    public class RestrictedBoltzmannMachineD : IRestrictedBoltzmannMachine<TElement>
    {
        private TElement[,] Weights;

        public RestrictedBoltzmannMachineD(int numVisible, int numHidden, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            NumHiddenNeurons = numHidden;
            NumVisibleNeurons = numVisible;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
            Weights = new TElement[numVisible + 1, numHidden + 1];
            Matrix2D.InsertValuesFrom(
                Weights,
                1,
                1,
                Matrix2D.Multiply(
                    Distributions.GaussianMatrixD(
                        numVisible,
                        numHidden),0.1));
        }

        public ActivationFunction VisibleActivation { get; protected set; }
        public ActivationFunction HiddenActivation { get; protected set; }


        public RestrictedBoltzmannMachineD(int numVisible, int numHidden, TElement[,] weights, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            NumHiddenNeurons = numHidden;
            NumVisibleNeurons = numVisible;
            Weights = weights;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
        }

        public int NumHiddenNeurons { get; protected set; }
        public int NumVisibleNeurons { get; protected set; }

        public TElement[,] Activate(TElement[,] matrix)
        {
            switch (VisibleActivation)
            {
                case ActivationFunction.Sigmoid:
                {
                    return ActivationFunctions.LogisticD(matrix);
                }
                case ActivationFunction.Tanh:
                {
                    return ActivationFunctions.TanhD(matrix);
                }
                case ActivationFunction.SoftPlus:
                {
                    return matrix;
                }
                default:
                    throw new NotImplementedException();
            }
        }

        public TElement[,] Encode(TElement[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);
            TElement[,] hiddenStates = Matrix2D.OnesD(numExamples, NumHiddenNeurons + 1);

            var data = new TElement[numExamples, visibleStates.GetLength(1) + 1];
            Matrix2D.InsertValuesFrom(data, 0, 1, visibleStates);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            TElement[,] hiddenActivations = Matrix2D.Multiply(data, Weights);

            TElement[,] hiddenProbs = Activate(hiddenActivations);
            hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                Distributions.UniformRandromMatrixD(numExamples, NumHiddenNeurons + 1));
            hiddenStates = Matrix2D.SubMatrix(hiddenStates, 0, 1);

            return hiddenStates;
        }

        public TElement[,] Decode(TElement[,] hiddenStates)
        {
            int numExamples = hiddenStates.GetLength(0);
            var data = new TElement[numExamples, hiddenStates.GetLength(1) + 1];

            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            Matrix2D.InsertValuesFrom(data, 0, 1, hiddenStates);

            TElement[,] visibleActivations = Matrix2D.Multiply(data, Matrix2D.Transpose(Weights));

            TElement[,] visibleProbs = Activate(visibleActivations);

            TElement[,] visibleStates = Matrix2D.GreaterThan(
                visibleProbs, Distributions.UniformRandromMatrixD(numExamples, NumVisibleNeurons + 1));

            visibleStates = Matrix2D.SubMatrix(visibleStates, 0, 1);
            return visibleStates;
        }

        public TElement[,] Reconstruct(TElement[,] data)
        {
            TElement[,] hidden = Encode(data);
            return Decode(hidden);
        }

        public TElement[,] DayDream(int numberOfSamples)
        {
            TElement[,] data = Matrix2D.OnesD(numberOfSamples, NumVisibleNeurons + 1);
            Matrix2D.InsertValuesFrom(data, 0, 1, Distributions.UniformRandromMatrixBoolD(1, NumVisibleNeurons), 1);
            //hiddenStates = Matrix2D.Update(hiddenStates, 0, 1, 1);
            for (int i = 0; i < numberOfSamples; i++)
            {
                TElement[,] visible = Matrix2D.SubMatrix(data, i, 0, 1);
                TElement[,] hiddenActivations = Matrix2D.ToVector(Matrix2D.Multiply(
                    visible, Weights));

                TElement[,] hiddenProbs = Activate(hiddenActivations);
                TElement[,] hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                    Distributions.UniformRandromVectorD(NumHiddenNeurons + 1));

                hiddenStates[0, 0] = 1;

                TElement[,] visibleActivations = Matrix2D.ToVector(Matrix2D.Multiply(hiddenStates, Matrix2D.Transpose(
                    Weights)));

                TElement[,] visibleProbs = Activate(visibleActivations);

                TElement[,] visibleStates = Matrix2D.GreaterThan(visibleProbs,
                    Distributions.UniformRandromVectorD(NumVisibleNeurons + 1));

                Matrix2D.InsertValuesFromRowOrColumn(data, visibleStates, 0, false, i, 0);
            }

            return Matrix2D.SubMatrix(data, 0, 1);
        }

        public void GreedyTrain(TElement[][] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            GreedyTrain(Matrix2D.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator, cancelToken);
        }

        public Task AsyncGreedyTrain(TElement[][] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            return AsyncGreedyTrain(Matrix2D.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator, cancelToken);
        }

        public void GreedyTrain(TElement[,] visibleData, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            exitEvaluator.Start();
            TElement error = 0d;

            int numExamples = visibleData.GetLength(0);
            var data = new TElement[numExamples, visibleData.GetLength(1) + 1];

            Matrix2D.InsertValuesFrom(data, 0, 1, visibleData);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);

            var sw = new Stopwatch();
            int i;
            for (i = 0;; i++)
            {
                cancelToken.ThrowIfCancellationRequested();
                sw.Start();

                TElement[,] posHiddenActivations = Matrix2D.Multiply(data, Weights);
                TElement[,] posHiddenProbs = Activate(posHiddenActivations);
                TElement[,] posHiddenStates = Matrix2D.GreaterThan(posHiddenProbs,
                    Distributions.UniformRandromMatrixD(numExamples, NumHiddenNeurons + 1));
                TElement[,] posAssociations = Matrix2D.Multiply(Matrix2D.Transpose(data), posHiddenProbs);

                TElement[,] negVisibleActivations = Matrix2D.Multiply(posHiddenStates, Matrix2D.Transpose(Weights));
                TElement[,] negVisibleProbs = Activate(negVisibleActivations);

                Matrix2D.UpdateValueAlongAxis(negVisibleProbs, 0, 1, Matrix2D.Axis.Vertical);

                TElement[,] negHiddenActivations = Matrix2D.Multiply(negVisibleProbs, Weights);
                TElement[,] negHiddenProbs = Activate(negHiddenActivations);

                TElement[,] negAssociations = Matrix2D.Multiply(Matrix2D.Transpose(negVisibleProbs), negHiddenProbs);

                Weights = Matrix2D.Add(
                    Weights,
                    Matrix2D.Multiply(
                        Matrix2D.Divide(
                            Matrix2D.Subtract(
                                posAssociations,
                                negAssociations),
                            numExamples),
                        learningRateCalculator.CalculateLearningRate(0, i)));

                error = Matrix2D.EnumerateElements(Matrix2D.Pow(Matrix2D.Subtract(data, negVisibleProbs), 2)).Sum();
                RaiseEpochEnd(i, error);


                double delta;
                if (exitEvaluator.Exit(i, error, sw.Elapsed, out delta))
                    break; 
                
                sw.Reset();
            }

            RaiseTrainEnd(i, error);
            exitEvaluator.Stop();
        }

        public Task AsyncGreedyTrain(TElement[,] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            return Task.Factory.StartNew(() => GreedyTrain(data, exitEvaluator, learningRateCalculator, cancelToken), cancelToken);
        }

        public event EventHandler<EpochEventArgs<TElement>> EpochEnd;

        public event EventHandler<EpochEventArgs<TElement>> TrainEnd;


        public ILayerSaveInfo<TElement> GetSaveInfo()
        {
            return new LayerSaveInfoD(NumVisibleNeurons, NumHiddenNeurons,
                Matrix2D.Duplicate(Weights, sizeof (TElement)), VisibleActivation, HiddenActivation);
        }



        private void RaiseTrainEnd(int epoch, TElement error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<TElement> {Epoch = epoch, Error = error});
        }

        private void RaiseEpochEnd(int epoch, TElement error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<TElement> {Epoch = epoch, Error = error});
        }


        public TElement CalculateReconstructionError(TElement[,] data)
        {
            throw new NotImplementedException();
        }
    }
}