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

        public RestrictedBoltzmannMachineD(int numVisible, int numHidden, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            NumHiddenNeurons = numHidden;
            NumVisibleNeurons = numVisible;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
            Weights = new double[numVisible + 1, numHidden + 1];
            Matrix2D.InsertValuesFrom(
                Weights,
                1,
                1,
                Matrix2D.Multiply(
                    Distributions.GaussianMatrix(
                        numVisible,
                        numHidden),0.1));
        }

        public ActivationFunction VisibleActivation { get; protected set; }
        public ActivationFunction HiddenActivation { get; protected set; }


        public RestrictedBoltzmannMachineD(int numVisible, int numHidden, double[,] weights, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            NumHiddenNeurons = numHidden;
            NumVisibleNeurons = numVisible;
            Weights = weights;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
        }

        public int NumHiddenNeurons { get; protected set; }
        public int NumVisibleNeurons { get; protected set; }

        public double[,] Activate(double[,] matrix)
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

        public double[,] Encode(double[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);
            double[,] hiddenStates = Matrix2D.OnesD(numExamples, NumHiddenNeurons + 1);

            var data = new double[numExamples, visibleStates.GetLength(1) + 1];
            Matrix2D.InsertValuesFrom(data, 0, 1, visibleStates);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            double[,] hiddenActivations = Matrix2D.Multiply(data, Weights);

            double[,] hiddenProbs = Activate(hiddenActivations);
            hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                Distributions.UniformRandromMatrixD(numExamples, NumHiddenNeurons + 1));
            hiddenStates = Matrix2D.SubMatrix(hiddenStates, 0, 1);

            return hiddenStates;
        }

        public double[,] Decode(double[,] hiddenStates)
        {
            int numExamples = hiddenStates.GetLength(0);
            var data = new double[numExamples, hiddenStates.GetLength(1) + 1];

            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            Matrix2D.InsertValuesFrom(data, 0, 1, hiddenStates);

            double[,] visibleActivations = Matrix2D.Multiply(data, Matrix2D.Transpose(Weights));

            double[,] visibleProbs = Activate(visibleActivations);

            double[,] visibleStates = Matrix2D.GreaterThan(
                visibleProbs, Distributions.UniformRandromMatrixD(numExamples, NumVisibleNeurons + 1));

            visibleStates = Matrix2D.SubMatrix(visibleStates, 0, 1);
            return visibleStates;
        }

        public double[,] Reconstruct(double[,] data)
        {
            double[,] hidden = Encode(data);
            return Decode(hidden);
        }

        public double[,] DayDream(int numberOfSamples)
        {
            double[,] data = Matrix2D.OnesD(numberOfSamples, NumVisibleNeurons + 1);
            Matrix2D.InsertValuesFrom(data, 0, 1, Distributions.UniformRandromMatrixBoolD(1, NumVisibleNeurons), 1);
            //hiddenStates = Matrix2D.Update(hiddenStates, 0, 1, 1);
            for (int i = 0; i < numberOfSamples; i++)
            {
                double[,] visible = Matrix2D.SubMatrix(data, i, 0, 1);
                double[,] hiddenActivations = Matrix2D.ToVector(Matrix2D.Multiply(
                    visible, Weights));

                double[,] hiddenProbs = Activate(hiddenActivations);
                double[,] hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                    Distributions.UniformRandromVectorD(NumHiddenNeurons + 1));

                hiddenStates[0, 0] = 1;

                double[,] visibleActivations = Matrix2D.ToVector(Matrix2D.Multiply(hiddenStates, Matrix2D.Transpose(
                    Weights)));

                double[,] visibleProbs = Activate(visibleActivations);

                double[,] visibleStates = Matrix2D.GreaterThan(visibleProbs,
                    Distributions.UniformRandromVectorD(NumVisibleNeurons + 1));

                Matrix2D.InsertValuesFromRowOrColumn(data, visibleStates, 0, false, i, 0);
            }

            return Matrix2D.SubMatrix(data, 0, 1);
        }

        public double GreedyTrain(double[][] data, IExitConditionEvaluator<double> exitEvaluator,
            ILearningRateCalculator<double> learningRateCalculator)
        {
            return GreedyTrain(Matrix2D.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator);
        }

        public Task<double> AsyncGreedyTrain(double[][] data, IExitConditionEvaluator<double> exitEvaluator,
            ILearningRateCalculator<double> learningRateCalculator)
        {
            return AsyncGreedyTrain(Matrix2D.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator);
        }

        public double GreedyTrain(double[,] visibleData, IExitConditionEvaluator<double> exitEvaluator,
            ILearningRateCalculator<double> learningRateCalculator)
        {
            exitEvaluator.Start();
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
                double[,] posHiddenProbs = Activate(posHiddenActivations);
                double[,] posHiddenStates = Matrix2D.GreaterThan(posHiddenProbs,
                    Distributions.UniformRandromMatrixD(numExamples, NumHiddenNeurons + 1));
                double[,] posAssociations = Matrix2D.Multiply(Matrix2D.Transpose(data), posHiddenProbs);

                double[,] negVisibleActivations = Matrix2D.Multiply(posHiddenStates, Matrix2D.Transpose(Weights));
                double[,] negVisibleProbs = Activate(negVisibleActivations);

                Matrix2D.UpdateValueAlongAxis(negVisibleProbs, 0, 1, Matrix2D.Axis.Vertical);

                double[,] negHiddenActivations = Matrix2D.Multiply(negVisibleProbs, Weights);
                double[,] negHiddenProbs = Activate(negHiddenActivations);

                double[,] negAssociations = Matrix2D.Multiply(Matrix2D.Transpose(negVisibleProbs), negHiddenProbs);

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

        public Task<double> AsyncGreedyTrain(double[,] data, IExitConditionEvaluator<double> exitEvaluator,
            ILearningRateCalculator<double> learningRateCalculator)
        {
            return Task.Run(() => GreedyTrain(data, exitEvaluator, learningRateCalculator));
        }

        public event EventHandler<EpochEventArgs<double>> EpochEnd;

        public event EventHandler<EpochEventArgs<double>> TrainEnd;


        public ILayerSaveInfo<double> GetSaveInfo()
        {
            return new LayerSaveInfoD(NumVisibleNeurons, NumHiddenNeurons,
                Matrix2D.Duplicate(Weights, sizeof (double)), VisibleActivation, HiddenActivation);
        }



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



        public double[,] GetSoftmaxLayer(double[,] visibleStates)
        {
            throw new NotImplementedException();
        }

        public double GreedySupervisedTrain(double[,] data, double[,] labels, IExitConditionEvaluator<double> exitEvaluator, ILearningRateCalculator<double> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public double GreedyBatchedSupervisedTrain(double[,] data, double[,] labels, int batchSize, IExitConditionEvaluator<double> exitEvaluator, ILearningRateCalculator<double> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public double[,] Classify(double[,] data, out double[,] labels)
        {
            throw new NotImplementedException();
        }

        public double GreedyBatchedTrain(double[][] data, int batchRows, IExitConditionEvaluator<double> exitEvaluator, ILearningRateCalculator<double> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public Task<double> AsyncGreedyBatchedTrain(double[][] data, int batchRows, IExitConditionEvaluator<double> exitEvaluator, ILearningRateCalculator<double> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public double GreedyBatchedTrain(double[,] data, int batchRows, IExitConditionEvaluator<double> exitEvaluator, ILearningRateCalculator<double> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public Task<double> AsyncGreedyBatchedTrain(double[,] data, int batchRows, IExitConditionEvaluator<double> exitEvaluator, ILearningRateCalculator<double> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public double CalculateReconstructionError(double[,] data)
        {
            throw new NotImplementedException();
        }
    }
}