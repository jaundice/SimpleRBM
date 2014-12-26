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
        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;

        public CudaRbmF(GPGPU gpu, GPGPURAND rand, int numVisible,
            int numHidden,
            IExitConditionEvaluator<float> exitCondition,
            float learningRate = 0.1f)
        {
            _gpu = gpu;
            _rand = rand;


            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;

            Console.WriteLine("Initializing {0}", LayerName);

            Matrix2D<float> weights = _gpu.AllocateAndSet<float>(numVisible + 1, numHidden + 1);

            using (Matrix2D<float> gaussian = GuassianDistribution(gpu, rand, numVisible, numHidden))
            using (Matrix2D<float> multGaussian = gaussian.Multiply(0.1f))
            {
                weights.InsertValuesFrom(1, 1, multGaussian);
                Weights = weights;
            }

            Console.WriteLine("Layer Initialized");
        }

        public CudaRbmF(GPGPU gpu, GPGPURAND rand, int numVisible, int numHidden, float[,] weights, IExitConditionEvaluator<float> exitCondition, float learningRate)
        {
            _gpu = gpu;
            _rand = rand;


            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;

            Matrix2D<float> gpuweights = _gpu.AllocateAndSet<float>(numVisible + 1, numHidden + 1);
            _gpu.CopyToDevice(weights, gpuweights);

            Weights = gpuweights;
        }

        private Matrix2D<float> Weights { get; set; }

        public string LayerName
        {
            get { return string.Format("Layer {0}x{1}", NumVisibleElements, NumHiddenElements); }
        }

        public float LearningRate { get; protected set; }
        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }
        public IExitConditionEvaluator<float> ExitConditionEvaluator { get; protected set; }

        public float[,] GetHiddenLayer(float[,] srcData)
        {
            int numExamples = srcData.GetLength(0);

            Matrix2D<float> tempSrcData = _gpu.AllocateAndSet<float>(srcData.GetLength(0), srcData.GetLength(1));
            _gpu.CopyToDevice(srcData, tempSrcData);

            Matrix2D<float> data = _gpu.AllocateAndSet<float>(numExamples, srcData.GetLength(1) + 1);

            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

            Matrix2D<float> hiddenActivations = data.Multiply(Weights);


            data.Dispose();

            Matrix2D<float> hiddenProbs = hiddenActivations.Logistic();

            hiddenActivations.Dispose();


            Matrix2D<float> uniformRand = UniformDistribution(_gpu, _rand, numExamples, NumHiddenElements + 1);

            Matrix2D<float> hsTemp = hiddenProbs.GreaterThan(uniformRand);

            hiddenProbs.Dispose();
            uniformRand.Dispose();

            Matrix2D<float> hiddenStates = hsTemp.SubMatrix(0, 1);

            hsTemp.Dispose();


            var localHiddenStates = new float[numExamples, NumHiddenElements];
            _gpu.CopyFromDevice(hiddenStates, localHiddenStates);

            hiddenStates.Dispose();

            return localHiddenStates;
        }

        public float[,] GetVisibleLayer(float[,] srcData)
        {
            int numExamples = srcData.GetLength(0);

            Matrix2D<float> data = _gpu.AllocateAndSet<float>(numExamples, srcData.GetLength(1) + 1);
            Matrix2D<float> tempSrcData = MatrixEx.Upload(_gpu, srcData);


            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            Matrix2D<float> transposedWeights = Weights.Transpose();

            Matrix2D<float> visibleActivations = data.Multiply(transposedWeights);


            data.Dispose();
            transposedWeights.Dispose();

            Matrix2D<float> visibleProbs = visibleActivations.Logistic();

            visibleActivations.Dispose();


            Matrix2D<float> randomDist = UniformDistribution(_gpu, _rand, numExamples, NumVisibleElements + 1);

            Matrix2D<float> visibleStatesTemp = visibleProbs.GreaterThan(randomDist);

            visibleProbs.Dispose();
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
            Matrix2D<float> data = _gpu.AllocateAndSet<float>(numberOfSamples, NumVisibleElements + 1);
            data.Ones();

            Matrix2D<float> uniform = UniformDistribution(_gpu, _rand, 1, NumVisibleElements);

            data.InsertValuesFrom(0, 1, uniform);

            uniform.Dispose();

            data.UpdateValuesAlongAxis(0, 1f, Axis.Row);


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

                hiddenStates.UpdateValuesAlongAxis(0, 0f, Axis.Column);

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

                data.InsertValuesFromRowOrColumn(visibleStates, 0, Axis.Row, i, 0);


                visibleStates.Dispose();
            }

            Matrix2D<float> returnVal = data.SubMatrix(0, 1);
            data.Dispose();
            float[,] localReturn = returnVal.CopyLocal();

            returnVal.Dispose();

            return localReturn;
        }

        public float Train(float[][] data)
        {
            return Train(Matrix2DCudaF.JaggedToMultidimesional(data));
        }

        public Task<float> AsyncTrain(float[][] data)
        {
            return AsyncTrain(Matrix2DCudaF.JaggedToMultidimesional(data));
        }

        public float Train(float[,] srcData)
        {
            ExitConditionEvaluator.Reset();
            float error = 0f;

            int numExamples = srcData.GetLength(0);
            int numCols = srcData.GetLength(1);
            int i;

            using (Matrix2D<float> data = _gpu.AllocateAndSet<float>(numExamples, numCols + 1))
            {
                using (Matrix2D<float> gpu_src = MatrixEx.Upload(_gpu, srcData))
                {
                    data.InsertValuesFrom(0, 1, gpu_src);
                    data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);
                }

                using (Matrix2D<float> dataTransposed = data.Transpose())
                {
                    var sw = new Stopwatch();
                    var errors = new List<float>();

                    _gpu.Synchronize();

                    for (i = 0; ; i++)
                    {
                        sw.Start();

                        Matrix2D<float> posHiddenActivations = data.Multiply(Weights);

                        Matrix2D<float> posHiddenProbs = posHiddenActivations.Logistic();


                        posHiddenActivations.Dispose();

                        Matrix2D<float> uniformRandom = UniformDistribution(_gpu, _rand, numExamples,
                            NumHiddenElements + 1);

                        Matrix2D<float> posHiddenStates = posHiddenProbs.GreaterThan(uniformRandom);

                        uniformRandom.Dispose();

                        Matrix2D<float> posAssociations = dataTransposed.Multiply(posHiddenProbs);

                        posHiddenProbs.Dispose();

                        Matrix2D<float> weightsTransposed = Weights.Transpose();

                        Matrix2D<float> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                        posHiddenStates.Dispose();
                        weightsTransposed.Dispose();

                        Matrix2D<float> negVisibleProbs = negVisibleActivations.Logistic();


                        negVisibleActivations.Dispose();

                        negVisibleProbs.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                        Matrix2D<float> negHiddenActivations = negVisibleProbs.Multiply(Weights);

                        Matrix2D<float> negHiddenProbs = negHiddenActivations.Logistic();

                        negHiddenActivations.Dispose();

                        Matrix2D<float> negVisibleProbsTransposed = negVisibleProbs.Transpose();

                        Matrix2D<float> negAssociations = negVisibleProbsTransposed.Multiply(negHiddenProbs);

                        negHiddenProbs.Dispose();
                        negVisibleProbsTransposed.Dispose();

                        Matrix2D<float> posAssocMinusNegAssoc = posAssociations.Subtract(negAssociations);

                        posAssociations.Dispose();
                        negAssociations.Dispose();

                        Matrix2D<float> tmult = posAssocMinusNegAssoc.Multiply(LearningRate / numExamples);

                        posAssocMinusNegAssoc.Dispose();

                        Matrix2D<float> tweight = Weights.Add(tmult);

                        Weights.Dispose();
                        tmult.Dispose();

                        Weights = tweight;

                        Matrix2D<float> delta = data.Subtract(negVisibleProbs);


                        negVisibleProbs.Dispose();

                        Matrix2D<float> pow = delta.Pow(2.0f);

                        delta.Dispose();

                        error = Sum(_gpu, pow, numExamples);

                        pow.Dispose();
                        errors.Add(error);
                        RaiseEpochEnd(i, error);

                        if (i % 20 == 0)
                            Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                                sw.ElapsedMilliseconds);
                        sw.Reset();

                        if (ExitConditionEvaluator.Exit(i, error))
                            break;

                    }
                }
            }

            RaiseTrainEnd(i, error);

            return error;
        }


        public Task<float> AsyncTrain(float[,] data)
        {
            return Task.Run(() => Train(data));
        }

        public event EventHandler<EpochEventArgs<float>> EpochEnd;

        public event EventHandler<EpochEventArgs<float>> TrainEnd;

        public static float Sum(GPGPU gpu, Matrix2D<float> matrix, int x)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(x, 1, out grid, out block);

            float[,] working = gpu.Allocate<float>(x, 1);
            gpu.Launch(grid, block, SumMatrixRowsF, matrix.Matrix, working);

            float[,] working2 = gpu.Allocate<float>(1, 1);
            gpu.Launch(new dim3(1), new dim3(1), SumMatrixColumnsF, working, working2);


            var local = new float[1, 1];
            gpu.CopyFromDevice(working2, local);

            gpu.Free(working);
            gpu.Free(working2);
            return local[0, 0];
        }

        [Cudafy]
        public static void SumMatrixRowsF(GThread thread, float[,] matrix, float[,] reduced)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (i < matrix.GetLength(0))
            {
                float sum = 0f;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j];
                }
                reduced[i, 0] = sum;
                i += thread.gridDim.x * thread.blockDim.x;
            }
        }

        [Cudafy]
        public static void SumMatrixColumnsF(GThread thread, float[,] matrix, float[,] reduced)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (i < matrix.GetLength(1))
            {
                float sum = 0f;
                for (int j = 0; j < matrix.GetLength(0); j++)
                {
                    sum += matrix[i, j];
                }
                reduced[0, i] = sum;
                i += thread.gridDim.x * thread.blockDim.x;
            }
        }

        public static Matrix2D<float> GuassianDistribution(GPGPU gpu, GPGPURAND rand, int x, int y)
        {
            Matrix2D<float> array = gpu.AllocateAndSet<float>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(1, y, out grid, out block);

            int my = (int)Math.Ceiling((float)y / 2) * 2;

            using (Matrix1D<float> tempGaussian = gpu.AllocateAndSet<float>(y))
            {
                for (int i = 0; i < x; i++)
                {
                    rand.GenerateNormal(tempGaussian, 0f, 1f, my);
                    gpu.Launch(grid, block, CopyToArrayAtNF, array.Matrix, tempGaussian.Matrix, i);
                }
            }
            return array;
        }


        public static Matrix2D<float> UniformDistribution(GPGPU gpu, GPGPURAND rand, int x, int y)
        {
            Matrix2D<float> array = gpu.AllocateAndSet<float>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(1, y, out grid, out block);
            int my = (int)Math.Ceiling((float)y / 2) * 2;

            using (Matrix1D<float> tempUniform = gpu.AllocateAndSet<float>(y))
            {
                for (int i = 0; i < x; i++)
                {
                    rand.GenerateUniform(tempUniform, my);

                    gpu.Launch(grid, block, CopyToArrayAtNF, array.Matrix, tempUniform.Matrix, i);
                }
            }
            return array;
        }


        [Cudafy]
        public static void CopyToArrayAtNF(GThread thread, float[,] target, float[] source, int x)
        {
            int id = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (id < source.GetLength(0))
            {
                target[x, id] = source[id];
                id += thread.blockDim.x * thread.gridDim.x;
            }

            thread.SyncThreads();
        }


        private void RaiseTrainEnd(int epoch, float error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<float> { Epoch = epoch, Error = error });
        }

        private void RaiseEpochEnd(int epoch, float error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<float> { Epoch = epoch, Error = error });
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


        public ILayerSaveInfo<float> GetSaveInfo()
        {
            return new LayerSaveInfoF(NumVisibleElements, NumHiddenElements, Weights.CopyLocal());
        }

        public void Dispose()
        {
            if (!Disposed)
            {
                this.Disposed = true;
                Dispose(true);
                GC.SuppressFinalize(this);
            }
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.Weights.Dispose();
            }
        }

        public bool Disposed { get; protected set; }

        ~CudaRbmF()
        {
            Dispose(false);
        }
    }
}