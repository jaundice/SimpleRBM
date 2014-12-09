using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;

namespace SimpleRBM.MultiDim
{
    public class RestrictedBoltzmannMachineD : IRestrictedBoltzmannMachine<double>
    {
        private double[,] Weights;

        public RestrictedBoltzmannMachineD(int numVisible, int numHidden, IExitConditionEvaluator<double> exitCondition,
            double learningRate = 0.1)
        {
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;

            Weights = new double[numVisible + 1, numHidden + 1];
            Matrix2D.InsertValuesFrom(
                Weights,
                1,
                1,
                Matrix2D.Multiply(
                    Distributions.GaussianMatrix(
                        numVisible,
                        numHidden),
                    learningRate));
        }


        public RestrictedBoltzmannMachineD(int numVisible, int numHidden, double[,] weights,
            IExitConditionEvaluator<double> exitCondition,
            double learningRate = 0.1)
        {
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;
            Weights = weights;
        }

        public double LearningRate { get; protected set; }
        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }

        public double[,] GetHiddenLayer(double[,] srcData)
        {
            int numExamples = srcData.GetLength(0);
            double[,] hiddenStates = Matrix2D.OnesD(numExamples, NumHiddenElements + 1);

            var data = new double[numExamples, srcData.GetLength(1) + 1];
            Matrix2D.InsertValuesFrom(data, 0, 1, srcData);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            double[,] hiddenActivations = Matrix2D.Multiply(data, Weights);

            double[,] hiddenProbs = ActivationFunctions.Logistic(hiddenActivations);
            hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                Distributions.UniformRandromMatrix(numExamples, NumHiddenElements + 1));
            hiddenStates = Matrix2D.SubMatrix(hiddenStates, 0, 1);

            return hiddenStates;
        }

        public double[,] GetVisibleLayer(double[,] srcData)
        {
            int numExamples = srcData.GetLength(0);
            var data = new double[numExamples, srcData.GetLength(1) + 1];
            Matrix2D.InsertValuesFrom(data, 0, 1, srcData);

            double[,] visibleActivations = Matrix2D.Multiply(data, Matrix2D.Transpose(Weights));

            double[,] visibleProbs = ActivationFunctions.Logistic(visibleActivations);

            double[,] visibleStates = Matrix2D.GreaterThan(
                visibleProbs, Distributions.UniformRandromMatrix(numExamples, NumVisibleElements + 1));

            visibleStates = Matrix2D.SubMatrix(visibleStates, 0, 1);
            return visibleStates;
        }

        public double[,] Reconstruct(double[,] data)
        {
            double[,] hidden = GetHiddenLayer(data);
            return GetVisibleLayer(hidden);
        }

        public double[,] DayDream(int numberOfSamples)
        {
            double[,] data = Matrix2D.OnesD(numberOfSamples, NumVisibleElements + 1);
            Matrix2D.InsertValuesFrom(data, 0, 1, Distributions.UniformRandromMatrixBool(1, NumVisibleElements), 1);
            //data = Matrix2D.Update(data, 0, 1, 1);
            for (int i = 0; i < numberOfSamples; i++)
            {
                double[,] visible = Matrix2D.SubMatrix(data, i, 0, 1);
                double[,] hiddenActivations = Matrix2D.ToVector(Matrix2D.Multiply(
                    visible, Weights));

                double[,] hiddenProbs = ActivationFunctions.Logistic(hiddenActivations);
                double[,] hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                    Distributions.UniformRandromVector(NumHiddenElements + 1));

                hiddenStates[0, 0] = 1;

                double[,] visibleActivations = Matrix2D.ToVector(Matrix2D.Multiply(hiddenStates, Matrix2D.Transpose(
                    Weights)));

                double[,] visibleProbs = ActivationFunctions.Logistic(visibleActivations);

                double[,] visibleStates = Matrix2D.GreaterThan(visibleProbs,
                    Distributions.UniformRandromVector(NumVisibleElements + 1));

                Matrix2D.InsertValuesFromRowOrColumn(data, visibleStates, 0, false, i, 0);
            }

            return Matrix2D.SubMatrix(data, 0, 1);
        }

        public double Train(double[][] data)
        {
            return Train(Matrix2D.JaggedToMultidimesional(data));
        }

        public Task<double> AsyncTrain(double[][] data)
        {
            return AsyncTrain(Matrix2D.JaggedToMultidimesional(data));
        }

        public double Train(double[,] srcData)
        {
            ExitConditionEvaluator.Reset();
            double error = 0d;

            int numExamples = srcData.GetLength(0);
            var data = new double[numExamples, srcData.GetLength(1) + 1];

            Matrix2D.InsertValuesFrom(data, 0, 1, srcData);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);

            var sw = new Stopwatch();
            var errors = new List<double>();
            int i;
            for (i = 0;; i++)
            {
                sw.Start();

                double[,] posHiddenActivations = Matrix2D.Multiply(data, Weights);
                double[,] posHiddenProbs = ActivationFunctions.Logistic(posHiddenActivations);
                double[,] posHiddenStates = Matrix2D.GreaterThan(posHiddenProbs,
                    Distributions.UniformRandromMatrix(numExamples, NumHiddenElements + 1));
                double[,] posAssociations = Matrix2D.Multiply(Matrix2D.Transpose(data), posHiddenProbs);

                double[,] negVisibleActivations = Matrix2D.Multiply(posHiddenStates, Matrix2D.Transpose(Weights));
                double[,] negVisibleProbs = ActivationFunctions.Logistic(negVisibleActivations);

                Matrix2D.UpdateValueAlongAxis(negVisibleProbs, 0, 1, Matrix2D.Axis.Vertical);

                double[,] negHiddenActivations = Matrix2D.Multiply(negVisibleProbs, Weights);
                double[,] negHiddenProbs = ActivationFunctions.Logistic(negHiddenActivations);

                double[,] negAssociations = Matrix2D.Multiply(Matrix2D.Transpose(negVisibleProbs), negHiddenProbs);

                Weights = Matrix2D.Add(
                    Weights,
                    Matrix2D.Multiply(
                        Matrix2D.Divide(
                            Matrix2D.Subtract(
                                posAssociations,
                                negAssociations),
                            numExamples),
                        LearningRate));

                error = Matrix2D.EnumerateElements(Matrix2D.Pow(Matrix2D.Subtract(data, negVisibleProbs), 2)).Sum();
                errors.Add(error);
                RaiseEpochEnd(i, error);

                if (i%20 == 0)
                    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                        sw.ElapsedMilliseconds);
                sw.Reset();


                if (ExitConditionEvaluator.Exit(i, error))
                    break;
            }

            RaiseTrainEnd(i, error);

            return error;
        }

        public Task<double> AsyncTrain(double[,] data)
        {
            return Task.Run(() => Train(data));
        }

        public event EventHandler<EpochEventArgs<double>> EpochEnd;

        public event EventHandler<EpochEventArgs<double>> TrainEnd;


        public ILayerSaveInfo<double> GetSaveInfo()
        {
            return new LayerSaveInfoD(NumVisibleElements, NumHiddenElements,
                Matrix2D.Duplicate(Weights, sizeof (double)));
        }


        public IExitConditionEvaluator<double> ExitConditionEvaluator { get; protected set; }

        private void RaiseTrainEnd(int epoch, double error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<double> {Epoch = epoch, Error = error});
        }

        private void RaiseEpochEnd(int epoch, double error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<double> {Epoch = epoch, Error = error});
        }
    }
}