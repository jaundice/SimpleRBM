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

namespace SimpleRBM.Cuda
{
    public class CudaRbmF : IRestrictedBoltzmannMachine<float>, IDisposable
    {
        private const float MOMENTUM = 0.4f;

        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;

        public CudaRbmF(GPGPU gpu, GPGPURAND rand, int numVisible,
            int numHidden,
            int layerIndex,
            IExitConditionEvaluator<float> exitCondition,
            ILearningRateCalculator<float> learningRate)
        {
            _gpu = gpu;
            _rand = rand;

            LayerIndex = layerIndex;
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;

            Console.WriteLine("Initializing {0}", LayerName);

            Matrix2D<float> weights = GuassianDistribution(gpu, rand, numVisible + 1, numHidden + 1, 0.1f);

            weights.UpdateValuesAlongAxis(0, 0f, Axis.Row);
            weights.UpdateValuesAlongAxis(0, 0f, Axis.Column);
            Weights = weights;

            IsInitialized = false;
            Console.WriteLine("Layer Initialized");
        }

        public CudaRbmF(GPGPU gpu, GPGPURAND rand, int numVisible, int numHidden, int layerIndex, float[,] weights,
            IExitConditionEvaluator<float> exitCondition, ILearningRateCalculator<float> learningRate)
        {
            _gpu = gpu;
            _rand = rand;

            LayerIndex = layerIndex;
            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;
            Matrix2D<float> gpuweights = _gpu.AllocateAndSet<float>(numVisible + 1, numHidden + 1);
            _gpu.CopyToDevice(weights, gpuweights);
            IsInitialized = true;

            Weights = gpuweights;
        }

        private bool IsInitialized { get; set; }

        private Matrix2D<float> Weights { get; set; }

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

        public ILearningRateCalculator<float> LearningRate { get; protected set; }
        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }
        public IExitConditionEvaluator<float> ExitConditionEvaluator { get; protected set; }


        public float[,] GetHiddenLayer(float[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);

            Matrix2D<float> tempSrcData = _gpu.Upload(visibleStates);

            Matrix2D<float> data = _gpu.AllocateNoSet<float>(numExamples, visibleStates.GetLength(1) + 1);

            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

            Matrix2D<float> hiddenActivations = data.Multiply(Weights);


            data.Dispose();

            hiddenActivations.LogisticInPlace();


            Matrix2D<float> uniformRand = UniformDistribution(_gpu, _rand, numExamples, NumHiddenElements + 1);

            Matrix2D<float> hsTemp = hiddenActivations.GreaterThan(uniformRand);

            hiddenActivations.Dispose();
            uniformRand.Dispose();

            Matrix2D<float> hiddenStates = hsTemp.SubMatrix(0, 1);

            hsTemp.Dispose();


            var localHiddenStates = new float[numExamples, NumHiddenElements];
            _gpu.CopyFromDevice(hiddenStates, localHiddenStates);

            hiddenStates.Dispose();

            return localHiddenStates;
        }

        public float[,] GetSoftmaxLayer(float[,] visibleStates)
        {
            int numExamples = visibleStates.GetLength(0);

            Matrix2D<float> tempSrcData = _gpu.AllocateNoSet<float>(visibleStates.GetLength(0),
                visibleStates.GetLength(1));
            _gpu.CopyToDevice(visibleStates, tempSrcData);

            Matrix2D<float> data = _gpu.AllocateNoSet<float>(numExamples, visibleStates.GetLength(1) + 1);

            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

            Matrix2D<float> hiddenActivations = data.Multiply(Weights);


            data.Dispose();


            //Matrix2D<float> fiftypc = _gpu.AllocateAndSet<float>(hiddenProbs.GetLength(0), hiddenProbs.GetLength(1));
            //fiftypc.Fill(0.5f);
            //Matrix2D<float> hsTemp = hiddenProbs.GreaterThan(fiftypc);

            //hiddenProbs.Dispose();
            //fiftypc.Dispose();

            //Matrix2D<float> hiddenStates = hsTemp.SubMatrix(0, 1);

            //hsTemp.Dispose();

            Matrix2D<float> hiddenStates = hiddenActivations.SubMatrix(0, 1);
            hiddenActivations.Dispose();


            Matrix2D<float> hiddenProbs = hiddenStates.SoftMax();

            Matrix2D<float> fiftypc = _gpu.AllocateAndSet<float>(hiddenProbs.GetLength(0), hiddenProbs.GetLength(1));
            fiftypc.Fill(0.5f);

            Matrix2D<float> hiddenStates2 = hiddenProbs.GreaterThan(fiftypc);

            hiddenProbs.Dispose();

            hiddenStates.Dispose();
            var localHiddenStates = new float[numExamples, NumHiddenElements];
            _gpu.CopyFromDevice(hiddenStates2, localHiddenStates);
            hiddenStates2.Dispose();

            return localHiddenStates;
        }

        public float[,] GetVisibleLayer(float[,] hiddenStates)
        {
            int numExamples = hiddenStates.GetLength(0);

            Matrix2D<float> data = _gpu.AllocateNoSet<float>(numExamples, hiddenStates.GetLength(1) + 1);
            using (Matrix2D<float> tempSrcData = _gpu.Upload(hiddenStates))
            {
                data.UpdateValuesAlongAxis(0, 1f, Axis.Column);
                data.InsertValuesFrom(0, 1, tempSrcData);

                tempSrcData.Dispose();
            }

            Matrix2D<float> transposedWeights = Weights.Transpose();

            Matrix2D<float> visibleActivations = data.Multiply(transposedWeights);


            data.Dispose();
            transposedWeights.Dispose();

            visibleActivations.LogisticInPlace();


            Matrix2D<float> randomDist = UniformDistribution(_gpu, _rand, numExamples, NumVisibleElements + 1);

            Matrix2D<float> visibleStatesTemp = visibleActivations.GreaterThan(randomDist);

            visibleActivations.Dispose();
            randomDist.Dispose();

            Matrix2D<float> visibleStates = visibleStatesTemp.SubMatrix(0, 1);


            visibleStatesTemp.Dispose();

            float[,] localVisStates = visibleStates.CopyLocal();

            visibleStates.Dispose();
            return localVisStates;
        }

        public float[,] Reconstruct(float[,] data)
        {
            float[,] hidden = GetHiddenLayer(data);
            return GetVisibleLayer(hidden);
        }

        public float[,] DayDream(int numberOfSamples)
        {
            Matrix2D<float> data = _gpu.AllocateNoSet<float>(numberOfSamples, NumVisibleElements + 1);
            //data.Ones();

            Matrix2D<float> uniform = UniformDistributionBool(_gpu, _rand, 1, NumVisibleElements);


            data.InsertValuesFrom(0, 1, uniform);

            data.UpdateValuesAlongAxis(0, 1f, Axis.Column);

            uniform.Dispose();

            //data.UpdateValuesAlongAxis(0, 1f, Axis.Row);


            for (int i = 0; i < numberOfSamples; i++)
            {
                Matrix2D<float> visible = data.SubMatrix(i, 0, 1, 0);

                Matrix2D<float> hiddenActivations = visible.Multiply(Weights);

                visible.Dispose();

                Matrix2D<float> hiddenProbs = hiddenActivations.Logistic();

                hiddenActivations.Dispose();

                Matrix2D<float> uniform2 = UniformDistribution(_gpu, _rand, 1, NumHiddenElements + 1);

                Matrix2D<float> hiddenStates = hiddenProbs.GreaterThan(uniform2);

                hiddenProbs.Dispose();
                uniform2.Dispose();

                hiddenStates.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                Matrix2D<float> weightsTransposed = Weights.Transpose();

                Matrix2D<float> visibleActivations = hiddenStates.Multiply(weightsTransposed);

                hiddenStates.Dispose();
                weightsTransposed.Dispose();

                Matrix2D<float> visibleProbs = visibleActivations.Logistic();

                visibleActivations.Dispose();


                Matrix2D<float> uniform3 = UniformDistribution(_gpu, _rand, 1, NumVisibleElements + 1);

                Matrix2D<float> visibleStates = visibleProbs.GreaterThan(uniform3);

                visibleProbs.Dispose();
                uniform3.Dispose();

                data.InsertValuesFromRowOrColumn(visibleStates, Axis.Row, i, 0);


                visibleStates.Dispose();
            }

            Matrix2D<float> returnVal = data.SubMatrix(0, 1);
            data.Dispose();
            float[,] localReturn = returnVal.CopyLocal();

            returnVal.Dispose();

            return localReturn;
        }

        public float GreedyTrain(float[][] data)
        {
            return GreedyTrain(Matrix2DCudaF.JaggedToMultidimesional(data));
        }

        public Task<float> AsyncGreedyTrain(float[][] data)
        {
            return AsyncGreedyTrain(Matrix2DCudaF.JaggedToMultidimesional(data));
        }

        public float CalculateReconstructionError(float[,] srcData)
        {
            float error = 0f;

            int numExamples = srcData.GetLength(0);
            int numCols = srcData.GetLength(1);
            int i;

            using (Matrix2D<float> data = _gpu.AllocateNoSet<float>(numExamples, numCols + 1))
            {
                using (Matrix2D<float> gpu_src = _gpu.Upload(srcData))
                {
                    data.InsertValuesFrom(0, 1, gpu_src);
                    data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);
                }

                _gpu.Synchronize();

                Matrix2D<float> posHiddenActivations = data.Multiply(Weights);

                posHiddenActivations.LogisticInPlace();


                Matrix2D<float> uniformRandom = UniformDistribution(_gpu, _rand, numExamples,
                    NumHiddenElements + 1);

                Matrix2D<float> posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);

                uniformRandom.Dispose();

                posHiddenActivations.Dispose();


                Matrix2D<float> weightsTransposed = Weights.Transpose();

                Matrix2D<float> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                posHiddenStates.Dispose();
                weightsTransposed.Dispose();

                Matrix2D<float> negVisibleProbs = negVisibleActivations.Logistic();


                negVisibleActivations.Dispose();

                negVisibleProbs.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                Matrix2D<float> delta = data.Subtract(negVisibleProbs);


                negVisibleProbs.Dispose();

                delta.PowInPlace(2.0f);


                error = Sum(_gpu, delta, numExamples);

                delta.Dispose();
            }
            return error;
        }

        public float GreedyTrain(float[,] visibleData)
        {
            ExitConditionEvaluator.Reset();
            float error = 0f;

            int numExamples = visibleData.GetLength(0);
            int numCols = visibleData.GetLength(1);
            int i;

            using (Matrix2D<float> data = _gpu.AllocateNoSet<float>(numExamples, numCols + 1))
            {
                using (Matrix2D<float> gpu_src = _gpu.Upload(visibleData))
                {
                    data.InsertValuesFrom(0, 1, gpu_src);
                    data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);
                }

                using (Matrix2D<float> dataTransposed = data.Transpose())
                {
                    var sw = new Stopwatch();

                    _gpu.Synchronize();

                    for (i = 0; ; i++)
                    {
                        sw.Start();

                        Matrix2D<float> posHiddenActivations = data.Multiply(Weights);

                        posHiddenActivations.LogisticInPlace();


                        Matrix2D<float> uniformRandom = UniformDistribution(_gpu, _rand, numExamples,
                            NumHiddenElements + 1);

                        Matrix2D<float> posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);

                        uniformRandom.Dispose();

                        Matrix2D<float> posAssociations = dataTransposed.Multiply(posHiddenActivations);

                        posHiddenActivations.Dispose();

                        Matrix2D<float> weightsTransposed = Weights.Transpose();

                        Matrix2D<float> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                        posHiddenStates.Dispose();
                        weightsTransposed.Dispose();

                        negVisibleActivations.LogisticInPlace();

                        negVisibleActivations.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                        Matrix2D<float> negHiddenActivations = negVisibleActivations.Multiply(Weights);

                        negHiddenActivations.LogisticInPlace();


                        Matrix2D<float> negVisibleProbsTransposed = negVisibleActivations.Transpose();

                        Matrix2D<float> negAssociations = negVisibleProbsTransposed.Multiply(negHiddenActivations);
                        negHiddenActivations.Dispose();

                        negVisibleProbsTransposed.Dispose();

                        posAssociations.SubtractInPlace(negAssociations);


                        negAssociations.Dispose();

                        posAssociations.MultiplyInPlace(LearningRate.CalculateLearningRate(LayerIndex, i) /
                                                        numExamples);

                        posAssociations.AddInPlace(Weights);

                        Weights.UpdateWithMomentumInPlace(posAssociations,
                            IsInitialized ? MOMENTUM : 1f);
                        IsInitialized = true;
                        posAssociations.Dispose();

                        Matrix2D<float> delta = data.Subtract(negVisibleActivations);


                        negVisibleActivations.Dispose();

                        delta.PowInPlace(2.0f);

                        error = Sum(_gpu, delta, numExamples);

                        delta.Dispose();
                        RaiseEpochEnd(i, error);

                        //if (i%20 == 0)
                        //    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                        //        sw.ElapsedMilliseconds);


                        if (ExitConditionEvaluator.Exit(i, error, sw.Elapsed))
                            break;
                        sw.Reset();
                    }
                }
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
            return new LayerSaveInfoF(NumVisibleElements, NumHiddenElements, Weights.CopyLocal());
        }

        public float GreedyBatchedTrain(float[][] data, int batchRows)
        {
            return GreedyBatchedTrain(Matrix2DCudaF.JaggedToMultidimesional(data), batchRows);
        }

        public Task<float> AsyncGreedyBatchedTrain(float[][] data, int batchRows)
        {
            return AsyncGreedyBatchedTrain(Matrix2DCudaF.JaggedToMultidimesional(data), batchRows);
        }

        public float GreedyBatchedTrain(float[,] srcData, int batchRows)
        {
            ExitConditionEvaluator.Reset();
            float error = 0f;

            //int numExamples = hiddenStates.GetLength(0);
            int numCols = srcData.GetLength(1);
            int i;

            var partitions = new List<Tuple<int, Matrix2D<float>>>();
            var transposedPartitions = new List<Matrix2D<float>>();
            using (Matrix2D<float> dataBlock = _gpu.AllocateNoSet<float>(srcData.GetLength(0), numCols + 1))
            {
                using (Matrix2D<float> gpu_src = _gpu.Upload(srcData))
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
                    Matrix2D<float> part = dataBlock.SubMatrix(j, 0, examples);

                    partitions.Add(Tuple.Create(examples, part));
                    transposedPartitions.Add(part.Transpose());
                }
            }

            var sw = new Stopwatch();
            var errors = new List<float>();

            _gpu.Synchronize();

            for (i = 0; ; i++)
            {
                sw.Start();
                int numExamples = partitions[i % partitions.Count].Item1;
                Matrix2D<float> data = partitions[i % partitions.Count].Item2;
                Matrix2D<float> dataTransposed = transposedPartitions[i % partitions.Count];

                Matrix2D<float> posHiddenActivations = data.Multiply(Weights);

                posHiddenActivations.LogisticInPlace();


                Matrix2D<float> uniformRandom = UniformDistribution(_gpu, _rand, numExamples,
                    NumHiddenElements + 1);

                Matrix2D<float> posHiddenStates = posHiddenActivations.GreaterThan(uniformRandom);

                uniformRandom.Dispose();

                Matrix2D<float> posAssociations = dataTransposed.Multiply(posHiddenActivations);

                posHiddenActivations.Dispose();

                Matrix2D<float> weightsTransposed = Weights.Transpose();

                Matrix2D<float> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                posHiddenStates.Dispose();
                weightsTransposed.Dispose();

                negVisibleActivations.LogisticInPlace();

                negVisibleActivations.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                Matrix2D<float> negHiddenActivations = negVisibleActivations.Multiply(Weights);

                negHiddenActivations.LogisticInPlace();


                Matrix2D<float> negVisibleProbsTransposed = negVisibleActivations.Transpose();

                Matrix2D<float> negAssociations = negVisibleProbsTransposed.Multiply(negHiddenActivations);
                negHiddenActivations.Dispose();

                negVisibleProbsTransposed.Dispose();

                Matrix2D<float> posAssocMinusNegAssoc = posAssociations.Subtract(negAssociations);

                posAssociations.Dispose();
                negAssociations.Dispose();

                posAssocMinusNegAssoc.MultiplyInPlace(LearningRate.CalculateLearningRate(LayerIndex, i) /
                                                      numExamples);

                posAssocMinusNegAssoc.AddInPlace(Weights);

                Weights.UpdateWithMomentumInPlace(posAssocMinusNegAssoc,
                    IsInitialized ? MOMENTUM : 1f);
                IsInitialized = true;
                posAssocMinusNegAssoc.Dispose();

                Matrix2D<float> delta = data.Subtract(negVisibleActivations);


                negVisibleActivations.Dispose();

                delta.PowInPlace(2.0f);

                error = Sum(_gpu, delta, numExamples);

                delta.Dispose();
                RaiseEpochEnd(i, error);

                //if (i%20 == 0)
                //    Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                //        sw.ElapsedMilliseconds);


                if (ExitConditionEvaluator.Exit(i, error, sw.Elapsed))
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

            return error;
        }

        public Task<float> AsyncGreedyBatchedTrain(float[,] data, int batchRows)
        {
            throw new NotImplementedException();
        }

        public float GreedySupervisedTrain(float[,] data, float[,] labels)
        {
            if (data.GetLength(0) != labels.GetLength(0))
            {
                throw new Exception("row count mismatch");
            }

            if (data.GetLength(1) + labels.GetLength(1) != NumVisibleElements)
            {
                throw new Exception("column count mismatch");
            }

            Matrix2D<float> dtmp = _gpu.Upload(data);
            Matrix2D<float> ltmp = _gpu.Upload(labels);
            Matrix2D<float> working = _gpu.AllocateAndSet<float>(data.GetLength(0), NumVisibleElements);
            working.InsertValuesFrom(0, 0, dtmp);
            working.InsertValuesFrom(0, dtmp.GetLength(1), ltmp);
            dtmp.Dispose();
            ltmp.Dispose();
            float[,] combined = working.CopyLocal();
            working.Dispose();

            return GreedyTrain(combined);
        }

        public float GreedyBatchedSupervisedTrain(float[,] data, float[,] labels, int batchSize)
        {
            if (data.GetLength(0) != labels.GetLength(0))
            {
                throw new Exception("row count mismatch");
            }

            if (data.GetLength(1) + labels.GetLength(1) != NumVisibleElements)
            {
                throw new Exception("column count mismatch");
            }

            float[,] combined;
            using (Matrix2D<float> dtmp = _gpu.Upload(data))
            {
                using (Matrix2D<float> ltmp = _gpu.Upload(labels))
                {
                    using (Matrix2D<float> working = _gpu.AllocateAndSet<float>(data.GetLength(0), NumVisibleElements))
                    {
                        working.InsertValuesFrom(0, 0, dtmp);
                        working.InsertValuesFrom(0, dtmp.GetLength(1), ltmp);

                        combined = working.CopyLocal();
                    }
                }
            }

            return GreedyBatchedTrain(combined, batchSize);
        }

        public float[,] Classify(float[,] data, out float[,] labels)
        {
            Matrix2D<float> dtmp = _gpu.Upload(data);
            Matrix2D<float> working = _gpu.AllocateAndSet<float>(data.GetLength(0), NumVisibleElements);
            working.InsertValuesFrom(0, 0, dtmp);
            dtmp.Dispose();
            float[,] combined = working.CopyLocal();
            working.Dispose();

            float[,] res = Reconstruct(combined);

            //float[,] soft = GetSoftmaxLayer(data);
            //float[,] res = GetVisibleLayer(soft);

            using (Matrix2D<float> res1 = _gpu.Upload(res))
            using (Matrix2D<float> dataT = res1.SubMatrix(0, 0, numCols: data.GetLength(1)))
            using (Matrix2D<float> label1 = res1.SubMatrix(0, data.GetLength(1)))
            {
                labels = label1.CopyLocal();
                return dataT.CopyLocal();
            }
        }

        public float[,] GetVisibleLayerLinear(float[,] hiddenData)
        {
            int numExamples = hiddenData.GetLength(0);

            Matrix2D<float> data = _gpu.AllocateNoSet<float>(numExamples, hiddenData.GetLength(1) + 1);
            using (Matrix2D<float> tempSrcData = _gpu.Upload(hiddenData))
            {
                data.UpdateValuesAlongAxis(0, 1f, Axis.Column);
                data.InsertValuesFrom(0, 1, tempSrcData);

                tempSrcData.Dispose();
            }

            Matrix2D<float> transposedWeights = Weights.Transpose();

            Matrix2D<float> visibleActivations = data.Multiply(transposedWeights);


            data.Dispose();
            transposedWeights.Dispose();

            visibleActivations.LogisticInPlace();


            Matrix2D<float> randomDist = UniformDistribution(_gpu, _rand, numExamples, NumVisibleElements + 1);

            Matrix2D<float> visibleStatesTemp = visibleActivations.GreaterThanLinear(randomDist);

            visibleActivations.Dispose();
            randomDist.Dispose();

            Matrix2D<float> visibleStates = visibleStatesTemp.SubMatrix(0, 1);


            visibleStatesTemp.Dispose();

            float[,] localVisStates = visibleStates.CopyLocal();

            visibleStates.Dispose();
            return localVisStates;
        }

        public void DownPass(float[,] hiddenStates, int epochsPerMachine, float learningRate, out float error)
        {
            error = float.MaxValue;
            //reconstruct visible
            int numExamples = hiddenStates.GetLength(0);


            using (Matrix2D<float> initialHiddenStates = _gpu.AllocateNoSet<float>(numExamples,
                hiddenStates.GetLength(1) + 1))
            {
                using (Matrix2D<float> tempSrcData = _gpu.Upload(hiddenStates))
                {
                    initialHiddenStates.InsertValuesFrom(0, 1, tempSrcData);
                    tempSrcData.Dispose();
                }
                initialHiddenStates.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                using (Matrix2D<float> initialHiddenStatesTransposed = initialHiddenStates.Transpose())
                {
                    for (int i = 0; i < epochsPerMachine; i++)
                    {
                        Matrix2D<float> transposedWeights = Weights.Transpose();

                        Matrix2D<float> posVisibleProbs = initialHiddenStates.Multiply(transposedWeights);


                        posVisibleProbs.LogisticInPlace();

                        posVisibleProbs.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                        Matrix2D<float> negHiddenProbs = posVisibleProbs.Multiply(Weights);

                        negHiddenProbs.LogisticInPlace();

                        Matrix2D<float> negVisibleProbs = initialHiddenStates.Multiply(transposedWeights);
                        transposedWeights.Dispose();
                        negVisibleProbs.LogisticInPlace();

                        var posVisibleProbsTransposed = posVisibleProbs.Transpose();
                        posVisibleProbs.Dispose();
                        var negVisibleProbsTransposed = negVisibleProbs.Transpose();

                        var posAssociations = posVisibleProbsTransposed.Multiply(initialHiddenStates);
                        var negAssociations = negVisibleProbsTransposed.Multiply(negHiddenProbs);

                        posVisibleProbsTransposed.Dispose();
                        negVisibleProbsTransposed.Dispose();
                        negVisibleProbs.Dispose();


                        Matrix2D<float> posMinusNegAssoc = posAssociations.Subtract(negAssociations);
                        posAssociations.Dispose();
                        negAssociations.Dispose();


                        posMinusNegAssoc.MultiplyInPlace(learningRate / numExamples);

                        posMinusNegAssoc.AddInPlace(Weights);
                        Weights.UpdateWithMomentumInPlace(posMinusNegAssoc, MOMENTUM);
                        posMinusNegAssoc.Dispose();


                        var err = initialHiddenStates.Subtract(negHiddenProbs);

                        err.PowInPlace(2f);
                        error = Sum(_gpu, err, numExamples);
                        err.Dispose();
                        negHiddenProbs.Dispose();

                        RaiseEpochEnd(i, error);
                        if (i % 20 == 0)
                            Console.WriteLine("Fine tune Layer:{0}\t\tepoch:{1}\t\terror:{2}", LayerIndex, i, error);
                    }
                    initialHiddenStatesTransposed.Dispose();
                }
                initialHiddenStates.Dispose();
            }
        }

        public static float Sum(GPGPU gpu, Matrix2D<float> matrix, int x)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(x, 1, out grid, out block);

            float[,] working = gpu.Allocate<float>(x, 1);
            gpu.Launch(grid, block, Matrix2DCudaF.SumMatrixRowsF, matrix.Matrix, working);

            float[,] working2 = gpu.Allocate<float>(1, 1);
            gpu.Launch(new dim3(1), new dim3(1), Matrix2DCudaF.SumMatrixColumnsF, working, working2);


            var local = new float[1, 1];
            gpu.CopyFromDevice(working2, local);

            gpu.Free(working);
            gpu.Free(working2);
            return local[0, 0];
        }

        public static Matrix2D<float> GuassianDistribution(GPGPU gpu, GPGPURAND rand, int x, int y, float scale = 1.0f)
        {
            Matrix2D<float> array = gpu.AllocateNoSet<float>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);

            using (Matrix1D<float> tempGaussian = gpu.AllocateNoSet<float>(x * y))
            {
                var len = x * y;
                if (len % 2 != 0)
                    len++;

                rand.GenerateNormal(tempGaussian, 0f, 0.5f, len);
                gpu.Launch(grid, block, Matrix2DCudaF.CopyToArrayAtNF2, array.Matrix, tempGaussian.Matrix, scale);
            }
            return array;
        }


        public static Matrix2D<float> UniformDistribution(GPGPU gpu, GPGPURAND rand, int x, int y, float scale = 1.0f)
        {
            Matrix2D<float> array = gpu.AllocateNoSet<float>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);

            using (Matrix1D<float> tempUniform = gpu.AllocateNoSet<float>(x * y))
            {
                rand.GenerateUniform(tempUniform, x * y);

                gpu.Launch(grid, block, Matrix2DCudaF.CopyToArrayAtNF2, array.Matrix, tempUniform.Matrix, scale);
            }
            return array;
        }

        public static Matrix2D<float> UniformDistributionBool(GPGPU gpu, GPGPURAND rand, int x, int y)
        {
            Matrix2D<float> array = UniformDistribution(gpu, rand, x, y);
            dim3 grid, block;

            ThreadOptimiser.Instance.GetStrategy(array, out grid, out block);
            gpu.Launch(grid, block, Matrix2DCudaF.ToBinaryF, array.Matrix);

            return array;
        }


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

        ~CudaRbmF()
        {
            Dispose(false);
        }
    }
}