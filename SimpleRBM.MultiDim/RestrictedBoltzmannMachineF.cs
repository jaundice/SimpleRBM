using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Markup;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;
using TElement = System.Single;
namespace SimpleRBM.MultiDim
{
    public class RestrictedBoltzmannMachineF : IRestrictedBoltzmannMachine<TElement>
    {
        private TElement[,] Weights;

        public RestrictedBoltzmannMachineF(int numVisible, int numHidden, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;

            Weights = new TElement[numVisible + 1, numHidden + 1];
            Matrix2D.InsertValuesFrom(
                Weights,
                1,
                1,
                Matrix2D.Multiply(
                    Distributions.GaussianMatrixF(
                        numVisible,
                        numHidden), 0.1f));
        }

        public ActivationFunction VisibleActivation { get; protected set; }
        public ActivationFunction HiddenActivation { get; protected set; }


        public RestrictedBoltzmannMachineF(int numVisible, int numHidden, TElement[,] weights, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            Weights = weights;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
        }

        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }

        public float[,] Activate(float[,] matrix)
        {
            switch (VisibleActivation)
            {
                case ActivationFunction.Sigmoid:
                    {
                        return ActivationFunctions.LogisticF(matrix);
                    }
                case ActivationFunction.Tanh:
                    {
                        return ActivationFunctions.TanhF(matrix);
                    }
                case ActivationFunction.SoftPlus:
                {
                    return matrix;
                }
                default:
                    throw new NotImplementedException();
            }
        }

        public TElement[,] GetHiddenLayer(TElement[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);
            TElement[,] hiddenStates = Matrix2D.OnesF(numExamples, NumHiddenElements + 1);

            var data = new TElement[numExamples, visibleStates.GetLength(1) + 1];
            Matrix2D.InsertValuesFrom(data, 0, 1, visibleStates);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            TElement[,] hiddenActivations = Matrix2D.Multiply(data, Weights);

            TElement[,] hiddenProbs = Activate(hiddenActivations);
            hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                Distributions.UniformRandromMatrixF(numExamples, NumHiddenElements + 1));
            hiddenStates = Matrix2D.SubMatrix(hiddenStates, 0, 1);

            return hiddenStates;
        }

        public TElement[,] GetVisibleLayer(TElement[,] hiddenStates)
        {
            int numExamples = hiddenStates.GetLength(0);
            var data = new TElement[numExamples, hiddenStates.GetLength(1) + 1];

            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            Matrix2D.InsertValuesFrom(data, 0, 1, hiddenStates);

            TElement[,] visibleActivations = Matrix2D.Multiply(data, Matrix2D.Transpose(Weights));

            TElement[,] visibleProbs = Activate(visibleActivations);

            TElement[,] visibleStates = Matrix2D.GreaterThan(
                visibleProbs, Distributions.UniformRandromMatrixF(numExamples, NumVisibleElements + 1));

            visibleStates = Matrix2D.SubMatrix(visibleStates, 0, 1);
            return visibleStates;
        }

        public TElement[,] Reconstruct(TElement[,] data)
        {
            TElement[,] hidden = GetHiddenLayer(data);
            return GetVisibleLayer(hidden);
        }

        public TElement[,] DayDream(int numberOfSamples)
        {
            TElement[,] data = Matrix2D.OnesF(numberOfSamples, NumVisibleElements + 1);
            Matrix2D.InsertValuesFrom(data, 0, 1, Distributions.UniformRandromMatrixBoolF(1, NumVisibleElements), 1);
            //hiddenStates = Matrix2D.Update(hiddenStates, 0, 1, 1);
            for (int i = 0; i < numberOfSamples; i++)
            {
                TElement[,] visible = Matrix2D.SubMatrix(data, i, 0, 1);
                TElement[,] hiddenActivations = Matrix2D.ToVector(Matrix2D.Multiply(
                    visible, Weights));

                TElement[,] hiddenProbs = Activate(hiddenActivations);
                TElement[,] hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                    Distributions.UniformRandromVectorF(NumHiddenElements + 1));

                hiddenStates[0, 0] = 1;

                TElement[,] visibleActivations = Matrix2D.ToVector(Matrix2D.Multiply(hiddenStates, Matrix2D.Transpose(
                    Weights)));

                TElement[,] visibleProbs = Activate(visibleActivations);

                TElement[,] visibleStates = Matrix2D.GreaterThan(visibleProbs,
                    Distributions.UniformRandromVectorF(NumVisibleElements + 1));

                Matrix2D.InsertValuesFromRowOrColumn(data, visibleStates, 0, false, i, 0);
            }

            return Matrix2D.SubMatrix(data, 0, 1);
        }

        public TElement GreedyTrain(TElement[][] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            return GreedyTrain(Matrix2D.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator);
        }

        public Task<TElement> AsyncGreedyTrain(TElement[][] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            return AsyncGreedyTrain(Matrix2D.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator);
        }

        public TElement GreedyTrain(TElement[,] visibleData, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            exitEvaluator.Start();
            TElement error = 0f;

            int numExamples = visibleData.GetLength(0);
            var data = new TElement[numExamples, visibleData.GetLength(1) + 1];

            Matrix2D.InsertValuesFrom(data, 0, 1, visibleData);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);

            var sw = new Stopwatch();
            int i;
            for (i = 0; ; i++)
            {
                sw.Start();

                TElement[,] posHiddenActivations = Matrix2D.Multiply(data, Weights);
                TElement[,] posHiddenProbs = Activate(posHiddenActivations);
                TElement[,] posHiddenStates = Matrix2D.GreaterThan(posHiddenProbs,
                    Distributions.UniformRandromMatrixF(numExamples, NumHiddenElements + 1));
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

                //if (i%20 == 0)
                //    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                //        sw.ElapsedMilliseconds);



                if (exitEvaluator.Exit(i, error, sw.Elapsed))
                    break;

                sw.Reset();
            }

            RaiseTrainEnd(i, error);
            exitEvaluator.Stop();
            return error;
        }

        public Task<TElement> AsyncGreedyTrain(TElement[,] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            return Task.Run(() => GreedyTrain(data, exitEvaluator, learningRateCalculator));
        }

        public event EventHandler<EpochEventArgs<TElement>> EpochEnd;

        public event EventHandler<EpochEventArgs<TElement>> TrainEnd;


        public ILayerSaveInfo<TElement> GetSaveInfo()
        {
            return new LayerSaveInfoF(NumVisibleElements, NumHiddenElements,
                Matrix2D.Duplicate(Weights, sizeof(TElement)), VisibleActivation, HiddenActivation);
        }



        private void RaiseTrainEnd(int epoch, TElement error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<TElement> { Epoch = epoch, Error = error });
        }

        private void RaiseEpochEnd(int epoch, TElement error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<TElement> { Epoch = epoch, Error = error });
        }



        public TElement[,] GetSoftmaxLayer(TElement[,] visibleStates)
        {
            throw new NotImplementedException();
        }

        public TElement GreedySupervisedTrain(TElement[,] data, TElement[,] labels, IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public TElement GreedyBatchedSupervisedTrain(TElement[,] data, TElement[,] labels, int batchSize, IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public TElement[,] Classify(TElement[,] data, out TElement[,] labels)
        {
            throw new NotImplementedException();
        }

        public TElement GreedyBatchedTrain(TElement[][] data, int batchRows, IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public Task<TElement> AsyncGreedyBatchedTrain(TElement[][] data, int batchRows, IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public TElement GreedyBatchedTrain(TElement[,] data, int batchRows, IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public Task<TElement> AsyncGreedyBatchedTrain(TElement[,] data, int batchRows, IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public TElement CalculateReconstructionError(TElement[,] data)
        {
            throw new NotImplementedException();
        }
    }
}