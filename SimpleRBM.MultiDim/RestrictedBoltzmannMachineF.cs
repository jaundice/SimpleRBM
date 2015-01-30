using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;

namespace SimpleRBM.MultiDim
{
    public class RestrictedBoltzmannMachineF : IRestrictedBoltzmannMachine<float>
    {
        private float[,] Weights;

        public RestrictedBoltzmannMachineF(int numVisible, int numHidden, int layerIndex, IExitConditionEvaluator<float> exitCondition,
            ILearningRateCalculator<float> learningRateCalculator)
        {
            LayerIndex = layerIndex;
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRateCalculator;

            Weights = new float[numVisible + 1, numHidden + 1];
            Matrix2D.InsertValuesFrom(
                Weights,
                1,
                1,
                Matrix2D.Multiply(
                    Distributions.GaussianMatrixF(
                        numVisible,
                        numHidden),
                    LearningRate.CalculateLearningRate(0, 0)));
        }

        public int LayerIndex { get; protected set; }


        public RestrictedBoltzmannMachineF(int numVisible, int numHidden, int layerIndex, float[,] weights,
            IExitConditionEvaluator<float> exitCondition,
            ILearningRateCalculator<float> learningRateCalculator)
        {
            LayerIndex = layerIndex;
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRateCalculator;
            Weights = weights;
        }

        public ILearningRateCalculator<float> LearningRate { get; protected set; }
        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }

        public float[,] GetHiddenLayer(float[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);
            float[,] hiddenStates = Matrix2D.OnesF(numExamples, NumHiddenElements + 1);

            var data = new float[numExamples, visibleStates.GetLength(1) + 1];
            Matrix2D.InsertValuesFrom(data, 0, 1, visibleStates);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            float[,] hiddenActivations = Matrix2D.Multiply(data, Weights);

            float[,] hiddenProbs = ActivationFunctions.LogisticF(hiddenActivations);
            hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                Distributions.UniformRandromMatrixF(numExamples, NumHiddenElements + 1));
            hiddenStates = Matrix2D.SubMatrix(hiddenStates, 0, 1);

            return hiddenStates;
        }

        public float[,] GetVisibleLayer(float[,] hiddenStates)
        {
            int numExamples = hiddenStates.GetLength(0);
            var data = new float[numExamples, hiddenStates.GetLength(1) + 1];
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);
            Matrix2D.InsertValuesFrom(data, 0, 1, hiddenStates);

            float[,] visibleActivations = Matrix2D.Multiply(data, Matrix2D.Transpose(Weights));

            float[,] visibleProbs = ActivationFunctions.LogisticF(visibleActivations);

            float[,] visibleStates = Matrix2D.GreaterThan(
                visibleProbs, Distributions.UniformRandromMatrixF(numExamples, NumVisibleElements + 1));

            visibleStates = Matrix2D.SubMatrix(visibleStates, 0, 1);
            return visibleStates;
        }

        public float[,] Reconstruct(float[,] data)
        {
            float[,] hidden = GetHiddenLayer(data);
            return GetVisibleLayer(hidden);
        }

        public float[,] DayDream(int numberOfSamples)
        {
            float[,] data = Matrix2D.OnesF(numberOfSamples, NumVisibleElements + 1);
            Matrix2D.InsertValuesFrom(data, 0, 1, Distributions.UniformRandromMatrixBoolF(1, NumVisibleElements), 1);
            //hiddenStates = Matrix2D.Update(hiddenStates, 0, 1, 1);
            for (int i = 0; i < numberOfSamples; i++)
            {
                float[,] visible = Matrix2D.SubMatrix(data, i, 0, 1);
                float[,] hiddenActivations = Matrix2D.ToVector(Matrix2D.Multiply(
                    visible, Weights));

                float[,] hiddenProbs = ActivationFunctions.LogisticF(hiddenActivations);
                float[,] hiddenStates = Matrix2D.GreaterThan(hiddenProbs,
                    Distributions.UniformRandromVectorF(NumHiddenElements + 1));

                hiddenStates[0, 0] = 1;

                float[,] visibleActivations = Matrix2D.ToVector(Matrix2D.Multiply(hiddenStates, Matrix2D.Transpose(
                    Weights)));

                float[,] visibleProbs = ActivationFunctions.LogisticF(visibleActivations);

                float[,] visibleStates = Matrix2D.GreaterThan(visibleProbs,
                    Distributions.UniformRandromVectorF(NumVisibleElements + 1));

                Matrix2D.InsertValuesFromRowOrColumn(data, visibleStates, 0, false, i, 0);
            }

            return Matrix2D.SubMatrix(data, 0, 1);
        }

        public float GreedyTrain(float[][] data)
        {
            return GreedyTrain(Matrix2D.JaggedToMultidimesional(data));
        }

        public Task<float> AsyncGreedyTrain(float[][] data)
        {
            return AsyncGreedyTrain(Matrix2D.JaggedToMultidimesional(data));
        }

        public float GreedyTrain(float[,] visibleData)
        {
            ExitConditionEvaluator.Reset();
            float error = 0f;

            int numExamples = visibleData.GetLength(0);
            var data = new float[numExamples, visibleData.GetLength(1) + 1];

            Matrix2D.InsertValuesFrom(data, 0, 1, visibleData);
            Matrix2D.UpdateValueAlongAxis(data, 0, 1, Matrix2D.Axis.Vertical);

            var sw = new Stopwatch();
            int i;
            for (i = 0; ; i++)
            {
                sw.Start();

                float[,] posHiddenActivations = Matrix2D.Multiply(data, Weights);
                float[,] posHiddenProbs = ActivationFunctions.LogisticF(posHiddenActivations);
                float[,] posHiddenStates = Matrix2D.GreaterThan(posHiddenProbs,
                    Distributions.UniformRandromMatrixF(numExamples, NumHiddenElements + 1));
                float[,] posAssociations = Matrix2D.Multiply(Matrix2D.Transpose(data), posHiddenProbs);

                float[,] negVisibleActivations = Matrix2D.Multiply(posHiddenStates, Matrix2D.Transpose(Weights));
                float[,] negVisibleProbs = ActivationFunctions.LogisticF(negVisibleActivations);

                Matrix2D.UpdateValueAlongAxis(negVisibleProbs, 0, 1, Matrix2D.Axis.Vertical);

                float[,] negHiddenActivations = Matrix2D.Multiply(negVisibleProbs, Weights);
                float[,] negHiddenProbs = ActivationFunctions.LogisticF(negHiddenActivations);

                float[,] negAssociations = Matrix2D.Multiply(Matrix2D.Transpose(negVisibleProbs), negHiddenProbs);

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

                //if (i % 20 == 0)
                //    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                //        sw.ElapsedMilliseconds);


                if (ExitConditionEvaluator.Exit(i, error, sw.Elapsed))
                    break;
                sw.Reset();
            }

            RaiseTrainEnd(i, error);

            return error;
        }

        public Task<float> AsyncGreedyTrain(float[,] data)
        {
            return Task.Run(() => GreedyTrain(data));
        }

        public event EventHandler<EpochEventArgs<float>> EpochEnd;

        public event EventHandler<EpochEventArgs<float>> TrainEnd;


        public ILayerSaveInfo<float> GetSaveInfo()
        {
            return new LayerSaveInfoF(NumVisibleElements, NumHiddenElements,
                Matrix2D.Duplicate(Weights, sizeof(float)));
        }


        public IExitConditionEvaluator<float> ExitConditionEvaluator { get; protected set; }

        private void RaiseTrainEnd(int epoch, float error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<float> { Layer = LayerIndex, Epoch = epoch, Error = error });
        }

        private void RaiseEpochEnd(int epoch, float error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<float> { Layer = LayerIndex, Epoch = epoch, Error = error });
        }


        public float GreedyBatchedTrain(float[][] data, int batchRows)
        {
            throw new NotImplementedException();
        }

        public Task<float> AsyncGreedyBatchedTrain(float[][] data, int batchRows)
        {
            throw new NotImplementedException();
        }

        public float GreedyBatchedTrain(float[,] data, int batchRows)
        {
            throw new NotImplementedException();
        }

        public Task<float> AsyncGreedyBatchedTrain(float[,] data, int batchRows)
        {
            throw new NotImplementedException();
        }


        public float CalculateReconstructionError(float[,] data)
        {
            throw new NotImplementedException();
        }


        public float[,] GetSoftmaxLayer(float[,] visibleStates)
        {
            throw new NotImplementedException();
        }


        public float GreedySupervisedTrain(float[,] data, float[,] labels)
        {
            throw new NotImplementedException();
        }

        public float[,] Classify(float[,] data, out float[,] labels)
        {
            throw new NotImplementedException();
        }


        public float GreedyBatchedSupervisedTrain(float[,] data, float[,] labels, int batchSize)
        {
            throw new NotImplementedException();
        }
    }
}