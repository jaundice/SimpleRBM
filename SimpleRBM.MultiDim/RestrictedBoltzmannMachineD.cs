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
            ILearningRateCalculator<double> learningRateCalculator)
        {
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRateCalculator;

            Weights = new double[numVisible + 1, numHidden + 1];
            Matrix2D.InsertValuesFrom(
                Weights,
                1,
                1,
                Matrix2D.Multiply(
                    Distributions.GaussianMatrix(
                        numVisible,
                        numHidden),
                    LearningRate.CalculateLearningRate(0,0)));
        }


        public RestrictedBoltzmannMachineD(int numVisible, int numHidden, double[,] weights,
            IExitConditionEvaluator<double> exitCondition,
            ILearningRateCalculator<double> learningRateCalculator )
        {
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRateCalculator;
            Weights = weights;
        }

        public ILearningRateCalculator<double> LearningRate { get; protected set; }
        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }

        public double[,] GetHiddenLayer(double[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);
            double[,] hiddenStates = Matrix2D.OnesD(numExamples, NumHiddenElements + 1);

            var data = new double[numExamples, visibleStates.GetLength(1) + 1];
            Matrix2D.InsertValuesFrom(data, 0, 1, visibleStates);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            double[,] hiddenActivations = Matrix2D.Multiply(data, Weights);

            double[,] hiddenProbs = ActivationFunctions.LogisticD(hiddenActivations);
            hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                Distributions.UniformRandromMatrixD(numExamples, NumHiddenElements + 1));
            hiddenStates = Matrix2D.SubMatrix(hiddenStates, 0, 1);

            return hiddenStates;
        }

        public double[,] GetVisibleLayer(double[,] hiddenStates)
        {
            int numExamples = hiddenStates.GetLength(0);
            var data = new double[numExamples, hiddenStates.GetLength(1) + 1];

            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            Matrix2D.InsertValuesFrom(data, 0, 1, hiddenStates);

            double[,] visibleActivations = Matrix2D.Multiply(data, Matrix2D.Transpose(Weights));

            double[,] visibleProbs = ActivationFunctions.LogisticD(visibleActivations);

            double[,] visibleStates = Matrix2D.GreaterThan(
                visibleProbs, Distributions.UniformRandromMatrixD(numExamples, NumVisibleElements + 1));

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
            Matrix2D.InsertValuesFrom(data, 0, 1, Distributions.UniformRandromMatrixBoolD(1, NumVisibleElements), 1);
            //hiddenStates = Matrix2D.Update(hiddenStates, 0, 1, 1);
            for (int i = 0; i < numberOfSamples; i++)
            {
                double[,] visible = Matrix2D.SubMatrix(data, i, 0, 1);
                double[,] hiddenActivations = Matrix2D.ToVector(Matrix2D.Multiply(
                    visible, Weights));

                double[,] hiddenProbs = ActivationFunctions.LogisticD(hiddenActivations);
                double[,] hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                    Distributions.UniformRandromVectorD(NumHiddenElements + 1));

                hiddenStates[0, 0] = 1;

                double[,] visibleActivations = Matrix2D.ToVector(Matrix2D.Multiply(hiddenStates, Matrix2D.Transpose(
                    Weights)));

                double[,] visibleProbs = ActivationFunctions.LogisticD(visibleActivations);

                double[,] visibleStates = Matrix2D.GreaterThan(visibleProbs,
                    Distributions.UniformRandromVectorD(NumVisibleElements + 1));

                Matrix2D.InsertValuesFromRowOrColumn(data, visibleStates, 0, false, i, 0);
            }

            return Matrix2D.SubMatrix(data, 0, 1);
        }

        public double GreedyTrain(double[][] data)
        {
            return GreedyTrain(Matrix2D.JaggedToMultidimesional(data));
        }

        public Task<double> AsyncGreedyTrain(double[][] data)
        {
            return AsyncGreedyTrain(Matrix2D.JaggedToMultidimesional(data));
        }

        public double GreedyTrain(double[,] visibleData)
        {
            ExitConditionEvaluator.Reset();
            double error = 0d;

            int numExamples = visibleData.GetLength(0);
            var data = new double[numExamples, visibleData.GetLength(1) + 1];

            Matrix2D.InsertValuesFrom(data, 0, 1, visibleData);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);

            var sw = new Stopwatch();
            int i;
            for (i = 0;; i++)
            {
                sw.Start();

                double[,] posHiddenActivations = Matrix2D.Multiply(data, Weights);
                double[,] posHiddenProbs = ActivationFunctions.LogisticD(posHiddenActivations);
                double[,] posHiddenStates = Matrix2D.GreaterThan(posHiddenProbs,
                    Distributions.UniformRandromMatrixD(numExamples, NumHiddenElements + 1));
                double[,] posAssociations = Matrix2D.Multiply(Matrix2D.Transpose(data), posHiddenProbs);

                double[,] negVisibleActivations = Matrix2D.Multiply(posHiddenStates, Matrix2D.Transpose(Weights));
                double[,] negVisibleProbs = ActivationFunctions.LogisticD(negVisibleActivations);

                Matrix2D.UpdateValueAlongAxis(negVisibleProbs, 0, 1, Matrix2D.Axis.Vertical);

                double[,] negHiddenActivations = Matrix2D.Multiply(negVisibleProbs, Weights);
                double[,] negHiddenProbs = ActivationFunctions.LogisticD(negHiddenActivations);

                double[,] negAssociations = Matrix2D.Multiply(Matrix2D.Transpose(negVisibleProbs), negHiddenProbs);

                Weights = Matrix2D.Add(
                    Weights,
                    Matrix2D.Multiply(
                        Matrix2D.Divide(
                            Matrix2D.Subtract(
                                posAssociations,
                                negAssociations),
                            numExamples),
                        LearningRate.CalculateLearningRate(0, i)));

                error = Matrix2D.EnumerateElements(Matrix2D.Pow(Matrix2D.Subtract(data, negVisibleProbs), 2)).Sum();
                RaiseEpochEnd(i, error);

                //if (i%20 == 0)
                //    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                //        sw.ElapsedMilliseconds);
               


                if (ExitConditionEvaluator.Exit(i, error, sw.Elapsed))
                    break; 
                
                sw.Reset();
            }

            RaiseTrainEnd(i, error);

            return error;
        }

        public Task<double> AsyncGreedyTrain(double[,] data)
        {
            return Task.Run(() => GreedyTrain(data));
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


        public double GreedyBatchedTrain(double[][] data, int batchRows)
        {
            throw new NotImplementedException();
        }

        public Task<double> AsyncGreedyBatchedTrain(double[][] data, int batchRows)
        {
            throw new NotImplementedException();
        }

        public double GreedyBatchedTrain(double[,] data, int batchRows)
        {
            throw new NotImplementedException();
        }

        public Task<double> AsyncGreedyBatchedTrain(double[,] data, int batchRows)
        {
            throw new NotImplementedException();
        }


        public double CalculateReconstructionError(double[,] data)
        {
            throw new NotImplementedException();
        }


        public double[,] GetSoftmaxLayer(double[,] visibleStates)
        {
            throw new NotImplementedException();
        }


        public double GreedySupervisedTrain(double[,] data, double[,] labels)
        {
            throw new NotImplementedException();
        }

        public double[,] Classify(double[,] data, out double[,] labels)
        {
            throw new NotImplementedException();
        }


        public double GreedyBatchedSupervisedTrain(double[,] data, double[,] labels, int batchSize)
        {
            throw new NotImplementedException();
        }
    }
}