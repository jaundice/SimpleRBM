//#define DEBUGCUDA

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Common.Save;
using TElement = System.Double;
using LSI = SimpleRBM.Common.Save.LayerSaveInfoD;

namespace SimpleRBM.Cuda
{
    public class CudaRbmD : IBasicRbmCuda<TElement>
    {
        private const TElement MOMENTUM = 0.9f;

        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;

        public CudaRbmD(GPGPU gpu, GPGPURAND rand, int numVisible,
            int numHidden,
            int layerIndex, ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            _gpu = gpu;
            _rand = rand;

            LayerIndex = layerIndex;
            NumHiddenNeurons = numHidden;
            NumVisibleNeurons = numVisible;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;

            Console.WriteLine("Initializing {0}", LayerName);

            Matrix2D<TElement> weights = gpu.GuassianDistribution(rand, numVisible + 1, numHidden + 1,
                (TElement)0,
                (TElement)0.5,
                (TElement)0.1);

            weights.UpdateValuesAlongAxis(0, 0f, Axis.Row);
            weights.UpdateValuesAlongAxis(0, 0f, Axis.Column);
            Weights = weights;

            IsInitialized = false;
            Console.WriteLine("Layer Initialized");
        }

        public CudaRbmD(GPGPU gpu, GPGPURAND rand, int numVisible, int numHidden, int layerIndex, TElement[,] weights,
            ActivationFunction visibleActivation, ActivationFunction hiddenActivation)
        {
            _gpu = gpu;
            _rand = rand;

            LayerIndex = layerIndex;
            NumHiddenNeurons = numHidden;
            NumVisibleNeurons = numVisible;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
            Matrix2D<TElement> gpuweights = GPU.AllocateAndSet<TElement>(numVisible + 1, numHidden + 1);
            GPU.CopyToDevice(weights, gpuweights.Matrix);
            IsInitialized = true;

            Weights = gpuweights;
        }

        private bool IsInitialized { get; set; }

        private Matrix2D<TElement> Weights { get; set; }

        public string LayerName
        {
            get { return string.Format("Layer {0}x{1}", NumVisibleNeurons, NumHiddenNeurons); }
        }

        public bool Disposed { get; protected set; }
        public int LayerIndex { get; protected set; }

        public void Dispose()
        {
            if (!Disposed)
            {
                Disposed = true;
                Dispose(true);
                GC.SuppressFinalize(this);
            }
        }

        public ActivationFunction VisibleActivation { get; protected set; }
        public ActivationFunction HiddenActivation { get; protected set; }

        public int NumHiddenNeurons { get; protected set; }
        public int NumVisibleNeurons { get; protected set; }

        public GPGPU GPU
        {
            get { return _gpu; }
        }

        public GPGPURAND GPURAND
        {
            get { return _rand; }
        }


        public TElement[,] Encode(TElement[,] visibleStates)
        {
            using (var data = GPU.Upload(visibleStates))
            using (var res = ((IBasicRbmCuda<TElement>)this).Encode(data))
            {
                return res.CopyLocal();
            }
        }

        public TElement[,] Decode(TElement[,] hiddenStates)
        {
            using (var data = GPU.Upload(hiddenStates))
            using (var res = ((IBasicRbmCuda<TElement>)this).Decode(data))
            {
                return res.CopyLocal();
            }
        }

        public TElement[,] Reconstruct(TElement[,] data)
        {
            using (var d = GPU.Upload(data))
            using (var res = ((IBasicRbmCuda<TElement>)this).Reconstruct(d))
            {
                return res.CopyLocal();
            }
        }

        public TElement[,] DayDream(int numberOfSamples)
        {
            using (var res = ((IBasicRbmCuda<TElement>)this).DayDream(numberOfSamples))
                return res.CopyLocal();
        }


        public TElement CalculateReconstructionError(TElement[,] srcData)
        {
            using (var data = GPU.Upload(srcData))
            {
                return ((IBasicRbmCuda<TElement>)this).CalculateReconstructionError(data);
            }
        }

        public void GreedyTrain(TElement[,] visibleData, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            using (var d = GPU.Upload(visibleData))
                ((IBasicRbmCuda<TElement>)this).GreedyTrain(d, exitEvaluator, learningRateCalculator, cancelToken);
        }


        public event EventHandler<EpochEventArgs<TElement>> EpochEnd;

        public event EventHandler<EpochEventArgs<TElement>> TrainEnd;

        public ILayerSaveInfo<TElement> GetSaveInfo()
        {
            return new LSI(NumVisibleNeurons, NumHiddenNeurons, Weights.CopyLocal(), VisibleActivation, HiddenActivation);
        }

        public TElement GreedyBatchedTrain(TElement[,] srcData, int batchRows,
            IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            using (var d = _gpu.Upload(srcData))
            {
                return ((IBasicRbmCuda<TElement>)this).GreedyBatchedTrain(d, batchRows, exitEvaluator,
                    learningRateCalculator, cancelToken);
            }
        }


        private void ActivateInPlace(Matrix2D<TElement> matrix, ActivationFunction activationFunction)
        {
            switch (activationFunction)
            {
                case ActivationFunction.Sigmoid:
                    {
                        matrix.LogisticInPlace();
                        break;
                    }
                case ActivationFunction.Tanh:
                    {
                        matrix.TanhInPlace();
                        break;
                    }
                case ActivationFunction.SoftPlus:
                    {
                        matrix.SoftPlusInPlace();
                        break;
                    }
                case ActivationFunction.SoftMax:
                    {
                        throw new NotImplementedException();
                    }
            }
        }


        void IBasicRbmCuda<TElement>.DownPass(Matrix2D<TElement> hiddenStates,
            IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, out TElement error, CancellationToken cancelToken)
        {
            error = TElement.MaxValue;
            //reconstruct visible
            int numExamples = hiddenStates.GetLength(0);

            exitEvaluator.Start();
            var sw = new Stopwatch();
            using (Matrix2D<TElement> initialHiddenStates = GPU.AllocateNoSet<TElement>(numExamples,
                hiddenStates.GetLength(1) + 1))
            {
                initialHiddenStates.InsertValuesFrom(0, 1, hiddenStates);

                initialHiddenStates.UpdateValuesAlongAxis(0, 1f, Axis.Column);


                for (int i = 0; ; i++)
                {
                    cancelToken.ThrowIfCancellationRequested();
                    sw.Restart();
                    Matrix2D<TElement> posVisibleProbs;
                    Matrix2D<TElement> negHiddenProbs;
                    Matrix2D<TElement> negVisibleProbs;
                    using (Matrix2D<TElement> transposedWeights = Weights.Transpose())
                    {
                        posVisibleProbs = initialHiddenStates.Multiply(transposedWeights);

                        ActivateInPlace(posVisibleProbs, VisibleActivation);

                        posVisibleProbs.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                        negHiddenProbs = posVisibleProbs.Multiply(Weights);

                        ActivateInPlace(negHiddenProbs, HiddenActivation);

                        negVisibleProbs = initialHiddenStates.Multiply(transposedWeights);
                    }
                    Matrix2D<TElement> posAssociations;
                    Matrix2D<TElement> negAssociations;
                    using (negVisibleProbs)
                    {
                        ActivateInPlace(negVisibleProbs, VisibleActivation);

                        using (posVisibleProbs)
                        using (Matrix2D<TElement> posVisibleProbsTransposed = posVisibleProbs.Transpose())
                        using (Matrix2D<TElement> negVisibleProbsTransposed = negVisibleProbs.Transpose())
                        {
                            posAssociations = posVisibleProbsTransposed.Multiply(initialHiddenStates);
                            negAssociations = negVisibleProbsTransposed.Multiply(negHiddenProbs);
                        }
                    }

                    using (posAssociations)
                    using (negAssociations)
                    using (Matrix2D<TElement> posMinusNegAssoc = posAssociations.Subtract(negAssociations))
                    {
                        posMinusNegAssoc.MultiplyInPlace(
                            learningRateCalculator.CalculateLearningRate(LayerIndex, i) /
                            numExamples);
                        posMinusNegAssoc.AddInPlace(Weights);
                        Weights.UpdateWithMomentumInPlace(posMinusNegAssoc, MOMENTUM);
                    }

                    using (negHiddenProbs)
                    using (Matrix2D<TElement> err = initialHiddenStates.Subtract(negHiddenProbs))
                    {
                        err.PowInPlace(2f);
                        error = err.Sum();
                    }

                    RaiseEpochEnd(i, error);
                    TElement delta;
                    if (exitEvaluator.Exit(i, error, sw.Elapsed, out delta))
                        break;
                }
            }

            exitEvaluator.Stop();
        }


        private void RaiseTrainEnd(int epoch, TElement error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<TElement> { Layer = LayerIndex, Epoch = epoch, Error = error });
        }

        private void RaiseEpochEnd(int epoch, TElement error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<TElement> { Layer = LayerIndex, Epoch = epoch, Error = error });
        }

        private static IEnumerable<T> EnumerateElements<T>(T[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    yield return matrix[i, j];
                }
            }
        }


        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                Weights.Dispose();
            }
        }

        ~CudaRbmD()
        {
            Dispose(false);
        }


        Matrix2D<TElement> IBasicRbmCuda<TElement>.Encode(Matrix2D<TElement> visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);

            using (Matrix2D<TElement> data = GPU.AllocateNoSet<TElement>(numExamples, visibleStates.GetLength(1) + 1))
            {
                data.InsertValuesFrom(0, 1, visibleStates);
                data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

                using (Matrix2D<TElement> hiddenActivations = data.Multiply(Weights))
                {
                    ActivateInPlace(hiddenActivations, HiddenActivation);
                    using (
                        Matrix2D<TElement> uniformRand = GPU.UniformDistribution(GPURAND, numExamples,
                            NumHiddenNeurons + 1, (TElement)1.0))

                    using (Matrix2D<TElement> hsTemp = hiddenActivations.GreaterThan(uniformRand))
                    {
                        hiddenActivations.Dispose();
                        return hsTemp.SubMatrix(0, 1);
                    }
                }
            }
        }

        Matrix2D<TElement> IBasicRbmCuda<TElement>.Decode(Matrix2D<TElement> activations)
        {
            int numExamples = activations.GetLength(0);
            using (Matrix2D<TElement> data = GPU.AllocateNoSet<TElement>(numExamples, activations.GetLength(1) + 1))
            using (Matrix2D<TElement> transposedWeights = Weights.Transpose())
            {
                data.UpdateValuesAlongAxis(0, 1f, Axis.Column);
                data.InsertValuesFrom(0, 1, activations);
                using (Matrix2D<TElement> visibleActivations = data.Multiply(transposedWeights))
                {
                    ActivateInPlace(visibleActivations, VisibleActivation);
                    using (
                        Matrix2D<TElement> randomDist = GPU.UniformDistribution(GPURAND, numExamples,
                            NumVisibleNeurons + 1, (TElement)1))
                    using (Matrix2D<TElement> visibleStatesTemp = visibleActivations.GreaterThan(randomDist))
                    {
                        return visibleStatesTemp.SubMatrix(0, 1);
                    }
                }
            }
        }

        Matrix2D<TElement> IBasicRbmCuda<TElement>.Reconstruct(Matrix2D<TElement> data)
        {
            return ((IBasicRbmCuda<TElement>)this).Decode(((IBasicRbmCuda<TElement>)this).Encode(data));
        }

        Matrix2D<TElement> IBasicRbmCuda<TElement>.DayDream(int numberOfSamples)
        {
            using (var rand = GPU.UniformDistribution(GPURAND, numberOfSamples, NumVisibleNeurons, (TElement)1))
            {
                return ((IBasicRbmCuda<TElement>)this).Reconstruct(rand);
            }
        }

        TElement IBasicRbmCuda<TElement>.GreedyBatchedTrain(Matrix2D<TElement> allData, int batchRows,
            IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            exitEvaluator.Start();
            TElement error = 0f;

            int numCols = allData.GetLength(1);
            int i;
            int numExamples = allData.GetLength(0);
            var partitions = new List<Tuple<Matrix2D<TElement>, Matrix2D<TElement>>>();
            var transposedPartitions = new List<Matrix2D<TElement>>();
            try
            {
                using (Matrix2D<TElement> dataBlock = GPU.AllocateNoSet<TElement>(allData.GetLength(0), numCols + 1))
                {
                    dataBlock.InsertValuesFrom(0, 1, allData);
                    dataBlock.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);


                    for (int j = 0; j < allData.GetLength(0); j += batchRows)
                    {
                        cancelToken.ThrowIfCancellationRequested();
                        int endIndex = j + batchRows;
                        if (endIndex > allData.GetLength(0) - 1)
                            endIndex = allData.GetLength(0) - 1;

                        int examples = endIndex - j;
                        Matrix2D<TElement> part = dataBlock.SubMatrix(j, 0, examples);

                        partitions.Add(Tuple.Create(part, part.Transpose()));
                    }
                }

                var sw = new Stopwatch();

                for (i = 0;; i++)
                {
                    cancelToken.ThrowIfCancellationRequested();
                    sw.Restart();
                    error =
                        partitions.Sum(
                            part => GreedyTrainInternal(part.Item1, part.Item2, i, numExamples, learningRateCalculator));

                    RaiseEpochEnd(i, error);

                    TElement delta;
                    if (exitEvaluator.Exit(i, error, sw.Elapsed, out delta))
                        break;
                }
            }
            finally
            {

                foreach (var partition in partitions)
                {
                    partition.Item2.Dispose();
                }
                foreach (var transposedPartition in transposedPartitions)
                {
                    transposedPartition.Dispose();
                }
            }
            RaiseTrainEnd(i, error);
            exitEvaluator.Stop();
            return error;
        }


        TElement IBasicRbmCuda<TElement>.GreedyTrain(Matrix2D<TElement> visibleData,
            IExitConditionEvaluator<TElement> exitEvaluator, ILearningRateCalculator<TElement> learningRateCalculator, CancellationToken cancelToken)
        {
            exitEvaluator.Start();
            TElement error = 0f;

            int numExamples = visibleData.GetLength(0);
            int numCols = visibleData.GetLength(1);
            int i;
            using (Matrix2D<TElement> data = GPU.AllocateNoSet<TElement>(numExamples, numCols + 1))
            {
                data.InsertValuesFrom(0, 1, visibleData);
                data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);


                using (Matrix2D<TElement> dataTransposed = data.Transpose())
                {
                    var sw = new Stopwatch();

                    for (i = 0; ; i++)
                    {
                        cancelToken.ThrowIfCancellationRequested();
                        sw.Restart();

                        error = GreedyTrainInternal(data, dataTransposed, i, numExamples, learningRateCalculator);

                        RaiseEpochEnd(i, error);

                        TElement delta;
                        if (exitEvaluator.Exit(i, error, sw.Elapsed, out delta))
                            break;
                    }
                }
            }

            RaiseTrainEnd(i, error);
            exitEvaluator.Stop();
            return error;
        }


        private TElement GreedyTrainInternal(Matrix2D<TElement> visibleDataWithBias,
            Matrix2D<TElement> visibleDataWithBiasTansposed, int epoch, int numExamples,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            TElement error = 0f;




            Matrix2D<TElement> posHiddenStates;
            Matrix2D<TElement> posAssociations;
            using (Matrix2D<TElement> posHiddenActivations = visibleDataWithBias.Multiply(Weights))
            {
                ActivateInPlace(posHiddenActivations, HiddenActivation);

                using (
                    Matrix2D<TElement> uniformRandom = GPU.UniformDistribution(GPURAND, numExamples,
                        NumHiddenNeurons + 1, (TElement)1))
                {
                    posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);
                }

                posAssociations = visibleDataWithBiasTansposed.Multiply(posHiddenActivations);
            }

            Matrix2D<TElement> negVisibleActivations;
            using (Matrix2D<TElement> weightsTransposed = Weights.Transpose())
            using (posHiddenStates)
            {
                negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);
                ActivateInPlace(negVisibleActivations, VisibleActivation);
                negVisibleActivations.UpdateValuesAlongAxis(0, 1f, Axis.Column);
            }


            Matrix2D<TElement> negAssociations;
            using (Matrix2D<TElement> negHiddenActivations = negVisibleActivations.Multiply(Weights))
            {
                using (Matrix2D<TElement> negVisibleProbsTransposed = negVisibleActivations.Transpose())
                {
                    ActivateInPlace(negHiddenActivations, HiddenActivation);
                    negAssociations = negVisibleProbsTransposed.Multiply(negHiddenActivations);
                }
            }

            using (posAssociations)
            using (negAssociations)
            {
                posAssociations.SubtractInPlace(negAssociations);


                posAssociations.MultiplyInPlace(learningRateCalculator.CalculateLearningRate(LayerIndex, epoch) /
                                                numExamples);

                posAssociations.AddInPlace(Weights);

                Weights.UpdateWithMomentumInPlace(posAssociations,
                    IsInitialized ? MOMENTUM : (TElement)0.5);
                if (epoch > 5)
                    IsInitialized = true;
            }

            using (Matrix2D<TElement> delta = visibleDataWithBias.Subtract(negVisibleActivations))
            using (negVisibleActivations)
            {
                delta.PowInPlace((TElement)2.0);

                error = delta.Sum();
            }

            return error;
        }


        TElement IBasicRbmCuda<TElement>.CalculateReconstructionError(Matrix2D<TElement> srcData)
        {
            TElement error;

            int numExamples = srcData.GetLength(0);
            int numCols = srcData.GetLength(1);
            int i;

            using (Matrix2D<TElement> data = GPU.AllocateNoSet<TElement>(numExamples, numCols + 1))
            {
                data.InsertValuesFrom(0, 1, srcData);
                data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

                Matrix2D<TElement> posHiddenStates;
                using (Matrix2D<TElement> posHiddenActivations = data.Multiply(Weights))
                {
                    ActivateInPlace(posHiddenActivations, HiddenActivation);

                    using (
                        Matrix2D<TElement> uniformRandom = GPU.UniformDistribution(GPURAND, numExamples,
                            NumHiddenNeurons + 1, (TElement)1))
                    {
                        posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);
                    }
                }


                Matrix2D<TElement> negVisibleActivations;
                using (Matrix2D<TElement> weightsTransposed = Weights.Transpose())
                using (posHiddenStates)
                {
                    negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);
                    ActivateInPlace(negVisibleActivations, VisibleActivation);
                    negVisibleActivations.UpdateValuesAlongAxis(0, 1f, Axis.Column);
                }


                using (negVisibleActivations)
                using (Matrix2D<TElement> delta = data.Subtract(negVisibleActivations))
                {
                    negVisibleActivations.Dispose();


                    delta.PowInPlace(2.0f);


                    error = delta.Sum(); //Sum(_gpu, delta, numExamples);
                }
            }
            return error;
        }
    }
}
