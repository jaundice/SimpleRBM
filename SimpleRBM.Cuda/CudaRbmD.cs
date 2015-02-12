//#define DEBUGCUDA

using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    public class CudaRbmD : IRestrictedBoltzmannMachine<TElement>, IDisposable
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
            //ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
            // LearningRate = learningRate;

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
            //ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            VisibleActivation = visibleActivation;
            HiddenActivation = hiddenActivation;
            //LearningRate = learningRate;
            Matrix2D<TElement> gpuweights = _gpu.AllocateAndSet<TElement>(numVisible + 1, numHidden + 1);
            _gpu.CopyToDevice(weights, gpuweights);
            IsInitialized = true;

            Weights = gpuweights;
        }

        private bool IsInitialized { get; set; }

        private Matrix2D<TElement> Weights { get; set; }

        public string LayerName
        {
            get { return string.Format("Layer {0}x{1}", NumVisibleElements, NumHiddenElements); }
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

        //public ILearningRateCalculator<TElement> LearningRate { get; protected set; }
        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }
        //public IExitConditionEvaluator<TElement> ExitConditionEvaluator { get; protected set; }


        public TElement[,] GetHiddenLayer(TElement[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);

            Matrix2D<TElement> tempSrcData = _gpu.Upload(visibleStates);

            Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numExamples, visibleStates.GetLength(1) + 1);

            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

            Matrix2D<TElement> hiddenActivations = data.Multiply(Weights);


            data.Dispose();

            ActivateInPlace(hiddenActivations, HiddenActivation);


            Matrix2D<TElement> uniformRand = _gpu.UniformDistribution(_rand, numExamples, NumHiddenElements + 1, (TElement)1.0);

            Matrix2D<TElement> hsTemp = hiddenActivations.GreaterThan(uniformRand);

            hiddenActivations.Dispose();
            uniformRand.Dispose();

            Matrix2D<TElement> hiddenStates = hsTemp.SubMatrix(0, 1);

            hsTemp.Dispose();


            var localHiddenStates = new TElement[numExamples, NumHiddenElements];
            _gpu.CopyFromDevice(hiddenStates, localHiddenStates);

            hiddenStates.Dispose();

            return localHiddenStates;
        }

        //public TElement[,] GetSoftmaxLayer(TElement[,] visibleStates)
        //{
        //    int numExamples = visibleStates.GetLength(0);

        //    Matrix2D<TElement> tempSrcData = _gpu.AllocateNoSet<TElement>(visibleStates.GetLength(0),
        //        visibleStates.GetLength(1));
        //    _gpu.CopyToDevice(visibleStates, tempSrcData);

        //    Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numExamples, visibleStates.GetLength(1) + 1);

        //    data.InsertValuesFrom(0, 1, tempSrcData);

        //    tempSrcData.Dispose();

        //    data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

        //    Matrix2D<TElement> hiddenActivations = data.Multiply(Weights);


        //    data.Dispose();


        //    //Matrix2D<TElement> fiftypc = _gpu.AllocateAndSet<TElement>(hiddenProbs.GetLength(0), hiddenProbs.GetLength(1));
        //    //fiftypc.Fill(0.5f);
        //    //Matrix2D<TElement> hsTemp = hiddenProbs.GreaterThan(fiftypc);

        //    //hiddenProbs.Dispose();
        //    //fiftypc.Dispose();

        //    //Matrix2D<TElement> hiddenStates = hsTemp.SubMatrix(0, 1);

        //    //hsTemp.Dispose();

        //    Matrix2D<TElement> hiddenStates = hiddenActivations.SubMatrix(0, 1);
        //    hiddenActivations.Dispose();


        //    Matrix2D<TElement> hiddenProbs = hiddenStates.SoftMax();

        //    Matrix2D<TElement> fiftypc = _gpu.AllocateAndSet<TElement>(hiddenProbs.GetLength(0), hiddenProbs.GetLength(1));
        //    fiftypc.Fill(0.5f);

        //    Matrix2D<TElement> hiddenStates2 = hiddenProbs.GreaterThan(fiftypc);

        //    hiddenProbs.Dispose();

        //    hiddenStates.Dispose();
        //    var localHiddenStates = new TElement[numExamples, NumHiddenElements];
        //    _gpu.CopyFromDevice(hiddenStates2, localHiddenStates);
        //    hiddenStates2.Dispose();

        //    return localHiddenStates;
        //}

        public TElement[,] GetVisibleLayer(TElement[,] hiddenStates)
        {
            int numExamples = hiddenStates.GetLength(0);

            Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numExamples, hiddenStates.GetLength(1) + 1);
            using (Matrix2D<TElement> tempSrcData = _gpu.Upload(hiddenStates))
            {
                data.UpdateValuesAlongAxis(0, 1f, Axis.Column);
                data.InsertValuesFrom(0, 1, tempSrcData);

                tempSrcData.Dispose();
            }

            Matrix2D<TElement> transposedWeights = Weights.Transpose();

            Matrix2D<TElement> visibleActivations = data.Multiply(transposedWeights);


            data.Dispose();
            transposedWeights.Dispose();

            //visibleActivations.LogisticInPlace();

            ActivateInPlace(visibleActivations, VisibleActivation);


            //Matrix2D<TElement> randomDist = UniformDistribution(_gpu, _rand, numExamples, NumVisibleElements + 1);

            //Matrix2D<TElement> visibleStatesTemp = visibleActivations.GreaterThan(randomDist);

            //visibleActivations.Dispose();
            //randomDist.Dispose();

            //Matrix2D<TElement> visibleStates = visibleStatesTemp.SubMatrix(0, 1);
            Matrix2D<TElement> visibleStates = visibleActivations.SubMatrix(0, 1);
            visibleActivations.Dispose();
            //visibleStatesTemp.Dispose();

            TElement[,] localVisStates = visibleStates.CopyLocal();

            visibleStates.Dispose();
            return localVisStates;
        }

        public TElement[,] Reconstruct(TElement[,] data)
        {
            TElement[,] hidden = GetHiddenLayer(data);
            return GetVisibleLayer(hidden);
        }

        public TElement[,] DayDream(int numberOfSamples)
        {
            Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numberOfSamples, NumVisibleElements + 1);
            //data.Ones();

            Matrix2D<TElement> uniform;
            _gpu.UniformDistributionBool(_rand, 1, NumVisibleElements, out uniform);


            data.InsertValuesFrom(0, 1, uniform);

            data.UpdateValuesAlongAxis(0, 1f, Axis.Column);

            uniform.Dispose();

            //data.UpdateValuesAlongAxis(0, 1f, Axis.Row);


            for (int i = 0; i < numberOfSamples; i++)
            {
                Matrix2D<TElement> visible = data.SubMatrix(i, 0, 1, 0);

                Matrix2D<TElement> hiddenActivations = visible.Multiply(Weights);

                visible.Dispose();

                //hiddenActivations.Logistic();
                ActivateInPlace(hiddenActivations, HiddenActivation);
                hiddenActivations.Dispose();

                Matrix2D<TElement> uniform2 = _gpu.UniformDistribution(_rand, 1, NumHiddenElements + 1, (TElement)1);

                Matrix2D<TElement> hiddenStates = hiddenActivations.GreaterThan(uniform2);

                hiddenActivations.Dispose();
                uniform2.Dispose();

                hiddenStates.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                Matrix2D<TElement> weightsTransposed = Weights.Transpose();

                Matrix2D<TElement> visibleActivations = hiddenStates.Multiply(weightsTransposed);

                hiddenStates.Dispose();
                weightsTransposed.Dispose();

                ActivateInPlace(visibleActivations, VisibleActivation);


                Matrix2D<TElement> uniform3 = _gpu.UniformDistribution(_rand, 1, NumVisibleElements + 1, (TElement)1);

                Matrix2D<TElement> visibleStates = visibleActivations.GreaterThan(uniform3);
                visibleActivations.Dispose();

                uniform3.Dispose();

                data.InsertValuesFromRowOrColumn(visibleStates, Axis.Row, i, 0);


                visibleStates.Dispose();
            }

            Matrix2D<TElement> returnVal = data.SubMatrix(0, 1);
            data.Dispose();
            TElement[,] localReturn = returnVal.CopyLocal();

            returnVal.Dispose();

            return localReturn;
        }

        public TElement GreedyTrain(TElement[][] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            return GreedyTrain(Matrix2DCuda.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator);
        }

        public Task<TElement> AsyncGreedyTrain(TElement[][] data, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            return AsyncGreedyTrain(Matrix2DCuda.JaggedToMultidimesional(data), exitEvaluator, learningRateCalculator);
        }

        public TElement CalculateReconstructionError(TElement[,] srcData)
        {
            TElement error = 0f;

            int numExamples = srcData.GetLength(0);
            int numCols = srcData.GetLength(1);
            int i;

            using (Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numExamples, numCols + 1))
            {
                using (Matrix2D<TElement> gpu_src = _gpu.Upload(srcData))
                {
                    data.InsertValuesFrom(0, 1, gpu_src);
                    data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);
                }

                _gpu.Synchronize();

                Matrix2D<TElement> posHiddenActivations = data.Multiply(Weights);

                ActivateInPlace(posHiddenActivations, HiddenActivation);
                //posHiddenActivations.LogisticInPlace();


                Matrix2D<TElement> uniformRandom = _gpu.UniformDistribution(_rand, numExamples, NumHiddenElements + 1, (TElement)1);

                Matrix2D<TElement> posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);

                uniformRandom.Dispose();

                posHiddenActivations.Dispose();


                Matrix2D<TElement> weightsTransposed = Weights.Transpose();

                Matrix2D<TElement> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                posHiddenStates.Dispose();
                weightsTransposed.Dispose();

                ActivateInPlace(negVisibleActivations, VisibleActivation);

                negVisibleActivations.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                Matrix2D<TElement> delta = data.Subtract(negVisibleActivations);

                negVisibleActivations.Dispose();


                delta.PowInPlace(2.0f);


                error = delta.Sum(); //Sum(_gpu, delta, numExamples);

                delta.Dispose();
            }
            return error;
        }

        public TElement GreedyTrain(TElement[,] visibleData, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            exitEvaluator.Start();
            TElement error = 0f;

            int numExamples = visibleData.GetLength(0);
            int numCols = visibleData.GetLength(1);
            int i;

            using (Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numExamples, numCols + 1))
            {
                using (Matrix2D<TElement> gpu_src = _gpu.Upload(visibleData))
                {
                    data.InsertValuesFrom(0, 1, gpu_src);
                    data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);
                }

                using (Matrix2D<TElement> dataTransposed = data.Transpose())
                {
                    var sw = new Stopwatch();

                    _gpu.Synchronize();

                    for (i = 0; ; i++)
                    {
                        sw.Start();

                        Matrix2D<TElement> posHiddenActivations = data.Multiply(Weights);

                        //posHiddenActivations.LogisticInPlace();
                        ActivateInPlace(posHiddenActivations, HiddenActivation);

                        Matrix2D<TElement> uniformRandom = _gpu.UniformDistribution(_rand, numExamples, NumHiddenElements + 1, (TElement)1);

                        Matrix2D<TElement> posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);

                        uniformRandom.Dispose();

                        Matrix2D<TElement> posAssociations = dataTransposed.Multiply(posHiddenActivations);

                        posHiddenActivations.Dispose();

                        Matrix2D<TElement> weightsTransposed = Weights.Transpose();

                        Matrix2D<TElement> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                        posHiddenStates.Dispose();
                        weightsTransposed.Dispose();

                        //negVisibleActivations.LogisticInPlace();
                        ActivateInPlace(negVisibleActivations, VisibleActivation);

                        negVisibleActivations.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                        Matrix2D<TElement> negHiddenActivations = negVisibleActivations.Multiply(Weights);

                        //negHiddenActivations.LogisticInPlace();
                        ActivateInPlace(negHiddenActivations, HiddenActivation);

                        Matrix2D<TElement> negVisibleProbsTransposed = negVisibleActivations.Transpose();

                        Matrix2D<TElement> negAssociations = negVisibleProbsTransposed.Multiply(negHiddenActivations);
                        negHiddenActivations.Dispose();

                        negVisibleProbsTransposed.Dispose();

                        posAssociations.SubtractInPlace(negAssociations);


                        negAssociations.Dispose();

                        posAssociations.MultiplyInPlace(learningRateCalculator.CalculateLearningRate(LayerIndex, i) /
                                                        numExamples);

                        posAssociations.AddInPlace(Weights);

                        Weights.UpdateWithMomentumInPlace(posAssociations,
                            IsInitialized ? MOMENTUM : 0.5f);
                        if (i > 5)
                            IsInitialized = true;

                        posAssociations.Dispose();

                        Matrix2D<TElement> delta = data.Subtract(negVisibleActivations);


                        negVisibleActivations.Dispose();

                        delta.PowInPlace(2.0f);

                        error = delta.Sum(); //Sum(_gpu, delta, numExamples);

                        delta.Dispose();
                        RaiseEpochEnd(i, error);

                        //if (i%20 == 0)
                        //    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                        //        sw.ElapsedMilliseconds);


                        if (exitEvaluator.Exit(i, error, sw.Elapsed))
                            break;
                        sw.Reset();
                    }
                }
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
            return new LSI(NumVisibleElements, NumHiddenElements, Weights.CopyLocal(), VisibleActivation, HiddenActivation);
        }

        public TElement GreedyBatchedTrain(TElement[][] data, int batchRows, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            return GreedyBatchedTrain(Matrix2DCuda.JaggedToMultidimesional(data), batchRows, exitEvaluator,
                learningRateCalculator);
        }

        public Task<TElement> AsyncGreedyBatchedTrain(TElement[][] data, int batchRows,
            IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            return AsyncGreedyBatchedTrain(Matrix2DCuda.JaggedToMultidimesional(data), batchRows, exitEvaluator,
                learningRateCalculator);
        }

        public TElement GreedyBatchedTrain(TElement[,] srcData, int batchRows, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            exitEvaluator.Start();
            TElement error = 0f;

            //int numExamples = hiddenStates.GetLength(0);
            int numCols = srcData.GetLength(1);
            int i;

            var partitions = new List<Tuple<int, Matrix2D<TElement>>>();
            var transposedPartitions = new List<Matrix2D<TElement>>();
            using (Matrix2D<TElement> dataBlock = _gpu.AllocateNoSet<TElement>(srcData.GetLength(0), numCols + 1))
            {
                using (Matrix2D<TElement> gpu_src = _gpu.Upload(srcData))
                {
                    dataBlock.InsertValuesFrom(0, 1, gpu_src);
                    dataBlock.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);
                }

                for (int j = 0; j < srcData.GetLength(0); j += batchRows)
                {
                    int endIndex = j + batchRows;
                    if (endIndex > srcData.GetLength(0) - 1)
                        endIndex = srcData.GetLength(0) - 1;

                    int examples = endIndex - j;
                    Matrix2D<TElement> part = dataBlock.SubMatrix(j, 0, examples);

                    partitions.Add(Tuple.Create(examples, part));
                    transposedPartitions.Add(part.Transpose());
                }
            }

            var sw = new Stopwatch();
            var errors = new List<TElement>();

            _gpu.Synchronize();

            for (i = 0; ; i++)
            {
                sw.Start();
                int numExamples = partitions[i % partitions.Count].Item1;
                Matrix2D<TElement> data = partitions[i % partitions.Count].Item2;
                Matrix2D<TElement> dataTransposed = transposedPartitions[i % partitions.Count];

                Matrix2D<TElement> posHiddenActivations = data.Multiply(Weights);

                //posHiddenActivations.LogisticInPlace();
                ActivateInPlace(posHiddenActivations, HiddenActivation);

                Matrix2D<TElement> uniformRandom = _gpu.UniformDistribution(_rand, numExamples, NumHiddenElements + 1, (TElement)1);

                Matrix2D<TElement> posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);

                uniformRandom.Dispose();

                Matrix2D<TElement> posAssociations = dataTransposed.Multiply(posHiddenActivations);

                posHiddenActivations.Dispose();

                Matrix2D<TElement> weightsTransposed = Weights.Transpose();

                Matrix2D<TElement> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                posHiddenStates.Dispose();
                weightsTransposed.Dispose();

                //negVisibleActivations.LogisticInPlace();
                ActivateInPlace(negVisibleActivations, VisibleActivation);

                negVisibleActivations.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                Matrix2D<TElement> negHiddenActivations = negVisibleActivations.Multiply(Weights);

                //negHiddenActivations.LogisticInPlace();
                ActivateInPlace(negHiddenActivations, HiddenActivation);


                Matrix2D<TElement> negVisibleProbsTransposed = negVisibleActivations.Transpose();

                Matrix2D<TElement> negAssociations = negVisibleProbsTransposed.Multiply(negHiddenActivations);
                negHiddenActivations.Dispose();

                negVisibleProbsTransposed.Dispose();

                Matrix2D<TElement> posAssocMinusNegAssoc = posAssociations.Subtract(negAssociations);

                posAssociations.Dispose();
                negAssociations.Dispose();

                posAssocMinusNegAssoc.MultiplyInPlace(learningRateCalculator.CalculateLearningRate(LayerIndex, i) /
                                                      numExamples);

                posAssocMinusNegAssoc.AddInPlace(Weights);

                Weights.UpdateWithMomentumInPlace(posAssocMinusNegAssoc,
                    IsInitialized ? MOMENTUM : 0.5f);
                if (i > 5)
                    IsInitialized = true;
                posAssocMinusNegAssoc.Dispose();

                Matrix2D<TElement> delta = data.Subtract(negVisibleActivations);


                negVisibleActivations.Dispose();

                delta.PowInPlace(2.0f);

                error = delta.Sum(); //Sum(_gpu, delta, numExamples);

                delta.Dispose();
                RaiseEpochEnd(i, error);

                //if (i%20 == 0)
                //    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                //        sw.ElapsedMilliseconds);


                if (exitEvaluator.Exit(i, error, sw.Elapsed))
                    break;
                sw.Reset();
            }


            foreach (var partition in partitions)
            {
                partition.Item2.Dispose();
            }
            foreach (var transposedPartition in transposedPartitions)
            {
                transposedPartition.Dispose();
            }

            RaiseTrainEnd(i, error);
            exitEvaluator.Stop();
            return error;
        }

        public Task<TElement> AsyncGreedyBatchedTrain(TElement[,] data, int batchRows,
            IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            throw new NotImplementedException();
        }

        public TElement GreedySupervisedTrain(TElement[,] data, TElement[,] labels, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            if (data.GetLength(0) != labels.GetLength(0))
            {
                throw new Exception("row count mismatch");
            }

            if (data.GetLength(1) + labels.GetLength(1) != NumVisibleElements)
            {
                throw new Exception("column count mismatch");
            }

            Matrix2D<TElement> dtmp = _gpu.Upload(data);
            Matrix2D<TElement> ltmp = _gpu.Upload(labels);
            Matrix2D<TElement> working = _gpu.AllocateAndSet<TElement>(data.GetLength(0), NumVisibleElements);
            working.InsertValuesFrom(0, 0, dtmp);
            working.InsertValuesFrom(0, dtmp.GetLength(1), ltmp);
            dtmp.Dispose();
            ltmp.Dispose();
            TElement[,] combined = working.CopyLocal();
            working.Dispose();

            return GreedyTrain(combined, exitEvaluator, learningRateCalculator);
        }

        public TElement GreedyBatchedSupervisedTrain(TElement[,] data, TElement[,] labels, int batchSize,
            IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator)
        {
            if (data.GetLength(0) != labels.GetLength(0))
            {
                throw new Exception("row count mismatch");
            }

            if (data.GetLength(1) + labels.GetLength(1) != NumVisibleElements)
            {
                throw new Exception("column count mismatch");
            }

            TElement[,] combined;
            using (Matrix2D<TElement> dtmp = _gpu.Upload(data))
            {
                using (Matrix2D<TElement> ltmp = _gpu.Upload(labels))
                {
                    using (Matrix2D<TElement> working = _gpu.AllocateAndSet<TElement>(data.GetLength(0), NumVisibleElements))
                    {
                        working.InsertValuesFrom(0, 0, dtmp);
                        working.InsertValuesFrom(0, dtmp.GetLength(1), ltmp);

                        combined = working.CopyLocal();
                    }
                }
            }

            return GreedyBatchedTrain(combined, batchSize, exitEvaluator, learningRateCalculator);
        }

        //public TElement[,] Classify(TElement[,] data, out TElement[,] labels)
        //{
        //    Matrix2D<TElement> dtmp = _gpu.Upload(data);
        //    Matrix2D<TElement> working = _gpu.AllocateAndSet<TElement>(data.GetLength(0), NumVisibleElements);
        //    working.InsertValuesFrom(0, 0, dtmp);
        //    dtmp.Dispose();
        //    TElement[,] combined = working.CopyLocal();
        //    working.Dispose();

        //    TElement[,] res = Reconstruct(combined);

        //    //TElement[,] soft = GetSoftmaxLayer(data);
        //    //TElement[,] res = GetVisibleLayer(soft);

        //    using (Matrix2D<TElement> res1 = _gpu.Upload(res))
        //    using (Matrix2D<TElement> dataT = res1.SubMatrix(0, 0, numCols: data.GetLength(1)))
        //    using (Matrix2D<TElement> label1 = res1.SubMatrix(0, data.GetLength(1)))
        //    {
        //        labels = label1.CopyLocal();
        //        return dataT.CopyLocal();
        //    }
        //}

        public TElement[,] Classify(TElement[,] visStates, out TElement[,] labels)
        {
            Matrix2D<TElement> dtmp = _gpu.Upload(visStates);
            Matrix2D<TElement> working = _gpu.AllocateAndSet<TElement>(visStates.GetLength(0), NumVisibleElements);
            working.InsertValuesFrom(0, 0, dtmp);
            dtmp.Dispose();
            TElement[,] combined = working.CopyLocal();
            working.Dispose();

            //TElement[,] res = Reconstruct(combined);

            TElement[,] hiddenStates = GetHiddenLayer(combined);

            //==================================

            int numExamples = hiddenStates.GetLength(0);

            Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numExamples, hiddenStates.GetLength(1) + 1);
            using (Matrix2D<TElement> tempSrcData = _gpu.Upload(hiddenStates))
            {
                data.UpdateValuesAlongAxis(0, 1f, Axis.Column);
                data.InsertValuesFrom(0, 1, tempSrcData);

                tempSrcData.Dispose();
            }

            Matrix2D<TElement> transposedWeights = Weights.Transpose();

            Matrix2D<TElement> visibleActivations = data.Multiply(transposedWeights);


            data.Dispose();
            transposedWeights.Dispose();

            //visibleActivations.LogisticInPlace();

            ActivateInPlace(visibleActivations, VisibleActivation);

            int maxLabels = NumVisibleElements - visStates.GetLength(1);

            using (Matrix2D<TElement> lbl = visibleActivations.SubMatrix(0, 1 + visStates.GetLength(1), 0, maxLabels))
            using (Matrix2D<TElement> rnd = _gpu.UniformDistribution(_rand, lbl.GetLength(0), lbl.GetLength(1), (TElement)1))
            using (Matrix2D<TElement> res = lbl.GreaterThan(rnd))
            //using (Matrix2D<TElement> sfmax = res.SoftMax())            
            {
                labels = res.CopyLocal();
            }


            Matrix2D<TElement> visstates = visibleActivations.SubMatrix(0, 1, 0, visStates.GetLength(1));


            Matrix2D<TElement> randomDist = _gpu.UniformDistribution(_rand, numExamples, visStates.GetLength(1), (TElement)1);

            Matrix2D<TElement> visibleStatesTemp = visstates.GreaterThan(randomDist);

            visibleActivations.Dispose();
            randomDist.Dispose();


            TElement[,] localVisStates = visibleStatesTemp.CopyLocal();

            visibleStatesTemp.Dispose();
            return localVisStates;


            //==================================

            //TElement[,] soft = GetSoftmaxLayer(data);
            //TElement[,] res = GetVisibleLayer(soft);

            //using (Matrix2D<TElement> res1 = _gpu.Upload(res))
            //using (Matrix2D<TElement> dataT = res1.SubMatrix(0, 0, numCols: data.GetLength(1)))
            //using (Matrix2D<TElement> label1 = res1.SubMatrix(0, data.GetLength(1)))
            //{
            //    labels = label1.CopyLocal();
            //    return dataT.CopyLocal();
            //}
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

        //public TElement[,] GetVisibleLayerLinear(TElement[,] hiddenData)
        //{
        //    int numExamples = hiddenData.GetLength(0);

        //    Matrix2D<TElement> data = _gpu.AllocateNoSet<TElement>(numExamples, hiddenData.GetLength(1) + 1);
        //    using (Matrix2D<TElement> tempSrcData = _gpu.Upload(hiddenData))
        //    {
        //        data.UpdateValuesAlongAxis(0, 1f, Axis.Column);
        //        data.InsertValuesFrom(0, 1, tempSrcData);

        //        tempSrcData.Dispose();
        //    }

        //    Matrix2D<TElement> transposedWeights = Weights.Transpose();

        //    Matrix2D<TElement> visibleActivations = data.Multiply(transposedWeights);


        //    data.Dispose();
        //    transposedWeights.Dispose();

        //    //visibleActivations.LogisticInPlace();
        //    ActivateInPlace(visibleActivations, VisibleActivation);


        //    Matrix2D<TElement> randomDist = UniformDistribution(_gpu, _rand, numExamples, NumVisibleElements + 1);

        //    Matrix2D<TElement> visibleStatesTemp = visibleActivations.GreaterThanLinear(randomDist);

        //    visibleActivations.Dispose();
        //    randomDist.Dispose();

        //    Matrix2D<TElement> visibleStates = visibleStatesTemp.SubMatrix(0, 1);


        //    visibleStatesTemp.Dispose();

        //    TElement[,] localVisStates = visibleStates.CopyLocal();

        //    visibleStates.Dispose();
        //    return localVisStates;
        //}

        public void DownPass(TElement[,] hiddenStates, IExitConditionEvaluator<TElement> exitEvaluator,
            ILearningRateCalculator<TElement> learningRateCalculator, out TElement error)
        {
            error = TElement.MaxValue;
            //reconstruct visible
            int numExamples = hiddenStates.GetLength(0);

            exitEvaluator.Start();
            var sw = new Stopwatch();
            using (Matrix2D<TElement> initialHiddenStates = _gpu.AllocateNoSet<TElement>(numExamples,
                hiddenStates.GetLength(1) + 1))
            {
                using (Matrix2D<TElement> tempSrcData = _gpu.Upload(hiddenStates))
                {
                    initialHiddenStates.InsertValuesFrom(0, 1, tempSrcData);
                    tempSrcData.Dispose();
                }
                initialHiddenStates.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                using (Matrix2D<TElement> initialHiddenStatesTransposed = initialHiddenStates.Transpose())
                {
                    for (int i = 0; ; i++)
                    {
                        sw.Restart();
                        Matrix2D<TElement> transposedWeights = Weights.Transpose();

                        Matrix2D<TElement> posVisibleProbs = initialHiddenStates.Multiply(transposedWeights);


                        //posVisibleProbs.LogisticInPlace();
                        ActivateInPlace(posVisibleProbs, VisibleActivation);

                        posVisibleProbs.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                        Matrix2D<TElement> negHiddenProbs = posVisibleProbs.Multiply(Weights);

                        //negHiddenProbs.LogisticInPlace();
                        ActivateInPlace(negHiddenProbs, HiddenActivation);

                        Matrix2D<TElement> negVisibleProbs = initialHiddenStates.Multiply(transposedWeights);
                        transposedWeights.Dispose();
                        //negVisibleProbs.LogisticInPlace();
                        ActivateInPlace(negVisibleProbs, VisibleActivation);

                        Matrix2D<TElement> posVisibleProbsTransposed = posVisibleProbs.Transpose();
                        posVisibleProbs.Dispose();
                        Matrix2D<TElement> negVisibleProbsTransposed = negVisibleProbs.Transpose();

                        Matrix2D<TElement> posAssociations = posVisibleProbsTransposed.Multiply(initialHiddenStates);
                        Matrix2D<TElement> negAssociations = negVisibleProbsTransposed.Multiply(negHiddenProbs);

                        posVisibleProbsTransposed.Dispose();
                        negVisibleProbsTransposed.Dispose();
                        negVisibleProbs.Dispose();


                        Matrix2D<TElement> posMinusNegAssoc = posAssociations.Subtract(negAssociations);
                        posAssociations.Dispose();
                        negAssociations.Dispose();


                        posMinusNegAssoc.MultiplyInPlace(learningRateCalculator.CalculateLearningRate(LayerIndex, i) /
                                                         numExamples);

                        posMinusNegAssoc.AddInPlace(Weights);
                        Weights.UpdateWithMomentumInPlace(posMinusNegAssoc, MOMENTUM);
                        posMinusNegAssoc.Dispose();


                        Matrix2D<TElement> err = initialHiddenStates.Subtract(negHiddenProbs);

                        err.PowInPlace(2f);
                        error = err.Sum();
                        err.Dispose();
                        negHiddenProbs.Dispose();

                        RaiseEpochEnd(i, error);
                        if (exitEvaluator.Exit(i, error, sw.Elapsed))
                            break;
                    }
                    initialHiddenStatesTransposed.Dispose();
                }
                initialHiddenStates.Dispose();
            }
            exitEvaluator.Stop();
        }

        //public static TElement Sum(GPGPU gpu, Matrix2D<TElement> matrix, int x)
        //{
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(x, 1, out grid, out block);

        //    TElement[,] working = gpu.Allocate<TElement>(x, 1);
        //    gpu.Launch(grid, block, Matrix2DCuda.SumMatrixRowsF, matrix.Matrix, working);

        //    TElement[,] working2 = gpu.Allocate<TElement>(1, 1);
        //    gpu.Launch(new dim3(1), new dim3(1), Matrix2DCuda.SumMatrixColumnsF, working, working2);


        //    var local = new TElement[1, 1];
        //    gpu.CopyFromDevice(working2, local);

        //    gpu.Free(working);
        //    gpu.Free(working2);
        //    return local[0, 0];
        //}

        //public static Matrix2D<TElement> GuassianDistribution(GPGPU gpu, GPGPURAND rand, int x, int y, TElement mean = 0f,
        //    TElement stDev = 0.5f, TElement scale = 1.0f)
        //{
        //    Matrix2D<TElement> array = gpu.AllocateNoSet<TElement>(x, y);
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);

        //    using (Matrix1D<TElement> tempGaussian = gpu.AllocateNoSet<TElement>(x * y))
        //    {
        //        int len = x * y;
        //        if (len % 2 != 0)
        //            len++;

        //        rand.GenerateNormal(tempGaussian, mean, stDev, len);
        //        gpu.Launch(grid, block, Matrix2DCuda.CopyToArrayAtNF2, array.Matrix, tempGaussian.Matrix, scale);
        //    }
        //    return array;
        //}


        //public static Matrix2D<TElement> UniformDistribution(GPGPU gpu, GPGPURAND rand, int x, int y, TElement scale = 1.0f)
        //{
        //    Matrix2D<TElement> array = gpu.AllocateNoSet<TElement>(x, y);
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);

        //    using (Matrix1D<TElement> tempUniform = gpu.AllocateNoSet<TElement>(x * y))
        //    {
        //        rand.GenerateUniform(tempUniform, x * y);

        //        gpu.Launch(grid, block, Matrix2DCuda.CopyToArrayAtNF2, array.Matrix, tempUniform.Matrix, scale);
        //    }
        //    return array;
        //}

        //public static Matrix2D<TElement> UniformDistributionBool(GPGPU gpu, GPGPURAND rand, int x, int y)
        //{
        //    Matrix2D<TElement> array = UniformDistribution(gpu, rand, x, y);
        //    dim3 grid, block;

        //    ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);
        //    gpu.Launch(grid, block, Matrix2DCuda.ToBinaryF, array.Matrix);

        //    return array;
        //}


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
    }
}
