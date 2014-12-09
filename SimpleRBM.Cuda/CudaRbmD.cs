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
    public class CudaRbmD : IRestrictedBoltzmannMachine<double>, IDisposable
    {
        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;

        public CudaRbmD(GPGPU gpu, GPGPURAND rand, int numVisible,
            int numHidden,
            IExitConditionEvaluator<double> exitCondition,
            double learningRate = 0.1)
        {
            _gpu = gpu;
            _rand = rand;


            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;

            Console.WriteLine("Initializing {0}", LayerName);

            Matrix2D<double> weights = _gpu.AllocateAndSet<double>(numVisible + 1, numHidden + 1);

            using (Matrix2D<double> gaussian = GuassianDistribution(gpu, rand, numVisible, numHidden))
            using (Matrix2D<double> multGaussian = gaussian.Multiply(0.1))
            {
                weights.InsertValuesFrom(1, 1, multGaussian);
                Weights = weights;
            }

            Console.WriteLine("Layer Initialized");
        }

        public CudaRbmD(GPGPU gpu, GPGPURAND rand, int numVisible, int numHidden, double[,] weights, IExitConditionEvaluator<double> exitCondition, double learningRate)
        {
            _gpu = gpu;
            _rand = rand;


            ExitConditionEvaluator = exitCondition;
            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;

            Matrix2D<double> gpuweights = _gpu.AllocateAndSet<double>(numVisible + 1, numHidden + 1);
            _gpu.CopyToDevice(weights, gpuweights);

            Weights = gpuweights;
        }

        private Matrix2D<double> Weights { get; set; }

        public string LayerName
        {
            get { return string.Format("Layer {0}x{1}", NumVisibleElements, NumHiddenElements); }
        }

        public double LearningRate { get; protected set; }
        public int NumHiddenElements { get; protected set; }
        public int NumVisibleElements { get; protected set; }
        public IExitConditionEvaluator<double> ExitConditionEvaluator { get; protected set; }

        public double[,] GetHiddenLayer(double[,] srcData)
        {
            int numExamples = srcData.GetLength(0);

            Matrix2D<double> tempSrcData = _gpu.AllocateAndSet<double>(srcData.GetLength(0), srcData.GetLength(1));
            _gpu.CopyToDevice(srcData, tempSrcData);

            Matrix2D<double> data = _gpu.AllocateAndSet<double>(numExamples, srcData.GetLength(1) + 1);

            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            data.UpdateValuesAlongAxis(0, 1.0, Axis.Column);

            Matrix2D<double> hiddenActivations = data.Multiply(Weights);


            data.Dispose();

            Matrix2D<double> hiddenProbs = hiddenActivations.Logistic();

            hiddenActivations.Dispose();


            Matrix2D<double> uniformRand = UniformDistribution(_gpu, _rand, numExamples, NumHiddenElements + 1);

            Matrix2D<double> hsTemp = hiddenProbs.GreaterThan(uniformRand);

            hiddenProbs.Dispose();
            uniformRand.Dispose();

            Matrix2D<double> hiddenStates = hsTemp.SubMatrix(0, 1);

            hsTemp.Dispose();


            var localHiddenStates = new double[numExamples, NumHiddenElements];
            _gpu.CopyFromDevice(hiddenStates, localHiddenStates);

            hiddenStates.Dispose();

            return localHiddenStates;
        }

        public double[,] GetVisibleLayer(double[,] srcData)
        {
            int numExamples = srcData.GetLength(0);

            Matrix2D<double> data = _gpu.AllocateAndSet<double>(numExamples, srcData.GetLength(1) + 1);
            Matrix2D<double> tempSrcData = MatrixEx.Upload(_gpu, srcData);


            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            Matrix2D<double> transposedWeights = Weights.Transpose();

            Matrix2D<double> visibleActivations = data.Multiply(transposedWeights);


            data.Dispose();
            transposedWeights.Dispose();

            Matrix2D<double> visibleProbs = visibleActivations.Logistic();

            visibleActivations.Dispose();


            Matrix2D<double> randomDist = UniformDistribution(_gpu, _rand, numExamples, NumVisibleElements + 1);

            Matrix2D<double> visibleStatesTemp = visibleProbs.GreaterThan(randomDist);

            visibleProbs.Dispose();
            randomDist.Dispose();

            Matrix2D<double> visibleStates = visibleStatesTemp.SubMatrix(0, 1);


            visibleStatesTemp.Dispose();

            double[,] localVisStates = visibleStates.CopyLocal();

            visibleStates.Dispose();
            return localVisStates;
        }

        public double[,] Reconstruct(double[,] data)
        {
            double[,] hidden = GetHiddenLayer(data);
            return GetVisibleLayer(hidden);
        }

        public double[,] DayDream(int numberOfSamples)
        {
            Matrix2D<double> data = _gpu.AllocateAndSet<double>(numberOfSamples, NumVisibleElements + 1);
            data.Ones();

            Matrix2D<double> uniform = UniformDistribution(_gpu, _rand, 1, NumVisibleElements);

            data.InsertValuesFrom(0, 1, uniform);

            uniform.Dispose();

            data.UpdateValuesAlongAxis(0, 1.0, Axis.Row);


            for (int i = 0; i < numberOfSamples; i++)
            {
                Matrix2D<double> visible = data.SubMatrix(i, 0, 1, 0);

                Matrix2D<double> hiddenActivations = visible.Multiply(Weights);

                visible.Dispose();

                Matrix2D<double> hiddenProbs = hiddenActivations.Logistic();

                hiddenActivations.Dispose();

                Matrix2D<double> uniform2 = UniformDistribution(_gpu, _rand, 1, NumHiddenElements + 1);

                Matrix2D<double> hiddenStates = hiddenProbs.GreaterThan(uniform2);

                hiddenProbs.Dispose();
                uniform2.Dispose();

                hiddenStates.UpdateValuesAlongAxis(0, 0.0, Axis.Column);

                Matrix2D<double> weightsTransposed = Weights.Transpose();

                Matrix2D<double> visibleActivations = hiddenStates.Multiply(weightsTransposed);

                hiddenStates.Dispose();
                weightsTransposed.Dispose();

                Matrix2D<double> visibleProbs = visibleActivations.Logistic();

                visibleActivations.Dispose();


                Matrix2D<double> uniform3 = UniformDistribution(_gpu, _rand, 1, NumVisibleElements + 1);

                Matrix2D<double> visibleStates = visibleProbs.GreaterThan(uniform3);

                visibleProbs.Dispose();
                uniform3.Dispose();

                data.InsertValuesFromRowOrColumn(visibleStates, 0, Axis.Row, i, 0);


                visibleStates.Dispose();
            }

            Matrix2D<double> returnVal = data.SubMatrix(0, 1);
            data.Dispose();
            double[,] localReturn = returnVal.CopyLocal();

            returnVal.Dispose();

            return localReturn;
        }

        public double Train(double[][] data)
        {
            return Train(Matrix2DCudaF.JaggedToMultidimesional(data));
        }

        public Task<double> AsyncTrain(double[][] data)
        {
            return AsyncTrain(Matrix2DCudaF.JaggedToMultidimesional(data));
        }

        public double Train(double[,] srcData)
        {
            ExitConditionEvaluator.Reset();
            double error = 0.0;

            int numExamples = srcData.GetLength(0);
            int numCols = srcData.GetLength(1);
            int i;

            using (Matrix2D<double> data = _gpu.AllocateAndSet<double>(numExamples, numCols + 1))
            {
                using (Matrix2D<double> gpu_src = MatrixEx.Upload(_gpu, srcData))
                {
                    data.InsertValuesFrom(0, 1, gpu_src);
                    data.UpdateValuesAlongAxis(0, 1.0, Axis.Column);
                }

                using (Matrix2D<double> dataTransposed = data.Transpose())
                {
                    var sw = new Stopwatch();
                    var errors = new List<double>();

                    _gpu.Synchronize();

                    for (i = 0; ; i++)
                    {
                        sw.Start();

                        Matrix2D<double> posHiddenActivations = data.Multiply(Weights);

                        Matrix2D<double> posHiddenProbs = posHiddenActivations.Logistic();


                        posHiddenActivations.Dispose();

                        Matrix2D<double> uniformRandom = UniformDistribution(_gpu, _rand, numExamples,
                            NumHiddenElements + 1);

                        Matrix2D<double> posHiddenStates = posHiddenProbs.GreaterThan(uniformRandom);

                        uniformRandom.Dispose();

                        Matrix2D<double> posAssociations = dataTransposed.Multiply(posHiddenProbs);

                        posHiddenProbs.Dispose();

                        Matrix2D<double> weightsTransposed = Weights.Transpose();

                        Matrix2D<double> negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                        posHiddenStates.Dispose();
                        weightsTransposed.Dispose();

                        Matrix2D<double> negVisibleProbs = negVisibleActivations.Logistic();


                        negVisibleActivations.Dispose();

                        negVisibleProbs.UpdateValuesAlongAxis(0, 1.0, Axis.Column);

                        Matrix2D<double> negHiddenActivations = negVisibleProbs.Multiply(Weights);

                        Matrix2D<double> negHiddenProbs = negHiddenActivations.Logistic();

                        negHiddenActivations.Dispose();

                        Matrix2D<double> negVisibleProbsTransposed = negVisibleProbs.Transpose();

                        Matrix2D<double> negAssociations = negVisibleProbsTransposed.Multiply(negHiddenProbs);

                        negHiddenProbs.Dispose();
                        negVisibleProbsTransposed.Dispose();

                        Matrix2D<double> posAssocMinusNegAssoc = posAssociations.Subtract(negAssociations);

                        posAssociations.Dispose();
                        negAssociations.Dispose();

                        Matrix2D<double> tmult = posAssocMinusNegAssoc.Multiply(LearningRate / numExamples);

                        posAssocMinusNegAssoc.Dispose();

                        Matrix2D<double> tweight = Weights.Add(tmult);

                        Weights.Dispose();
                        tmult.Dispose();

                        Weights = tweight;

                        Matrix2D<double> delta = data.Subtract(negVisibleProbs);


                        negVisibleProbs.Dispose();

                        Matrix2D<double> pow = delta.Pow(2.0);

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


        public Task<double> AsyncTrain(double[,] data)
        {
            return Task.Run(() => Train(data));
        }

        public event EventHandler<EpochEventArgs<double>> EpochEnd;

        public event EventHandler<EpochEventArgs<double>> TrainEnd;

        public static double Sum(GPGPU gpu, Matrix2D<double> matrix, int x)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(x, 1, out grid, out block);

            double[,] working = gpu.Allocate<double>(x, 1);
            gpu.Launch(grid, block, SumMatrixRowsD, matrix.Matrix, working);

            double[,] working2 = gpu.Allocate<double>(1, 1);
            gpu.Launch(new dim3(1), new dim3(1), SumMatrixColumnsD, working, working2);


            var local = new double[1, 1];
            gpu.CopyFromDevice(working2, local);

            gpu.Free(working);
            gpu.Free(working2);
            return local[0, 0];
        }

        [Cudafy]
        public static void SumMatrixRowsD(GThread thread, double[,] matrix, double[,] reduced)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (i < matrix.GetLength(0))
            {
                double sum = 0.0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j];
                }
                reduced[i, 0] = sum;
                i += thread.gridDim.x * thread.blockDim.x;
            }
        }

        [Cudafy]
        public static void SumMatrixColumnsD(GThread thread, double[,] matrix, double[,] reduced)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (i < matrix.GetLength(1))
            {
                double sum = 0.0;
                for (int j = 0; j < matrix.GetLength(0); j++)
                {
                    sum += matrix[i, j];
                }
                reduced[0, i] = sum;
                i += thread.gridDim.x * thread.blockDim.x;
            }
        }

        public static Matrix2D<double> GuassianDistribution(GPGPU gpu, GPGPURAND rand, int x, int y)
        {
            Matrix2D<double> array = gpu.AllocateAndSet<double>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(1, y, out grid, out block);

            int my = (int)Math.Ceiling((double)y / 2) * 2;

            using (Matrix1D<double> tempGaussian = gpu.AllocateAndSet<double>(y))
            {
                for (int i = 0; i < x; i++)
                {
                    rand.GenerateNormal(tempGaussian, 0.0f, 1.0f, my);
                    gpu.Launch(grid, block, CopyToArrayAtND, array.Matrix, tempGaussian.Matrix, i);
                }
            }
            return array;
        }


        public static Matrix2D<double> UniformDistribution(GPGPU gpu, GPGPURAND rand, int x, int y)
        {

            Matrix2D<double> array = gpu.AllocateAndSet<double>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(1, y, out grid, out block);
            int my = (int)Math.Ceiling((double)y / 2) * 2;

            using (Matrix1D<double> tempUniform = gpu.AllocateAndSet<double>(y))
            {
                for (int i = 0; i < x; i++)
                {
                    rand.GenerateUniform(tempUniform, my);

                    gpu.Launch(grid, block, CopyToArrayAtND, array.Matrix, tempUniform.Matrix, i);
                }
            }
            return array;
        }


        [Cudafy]
        public static void CopyToArrayAtND(GThread thread, double[,] target, double[] source, int x)
        {
            int id = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;

            while (id < source.GetLength(0))
            {
                target[x, id] = source[id];
                id += thread.blockDim.x * thread.gridDim.x;
            }

            thread.SyncThreads();
        }


        private void RaiseTrainEnd(int epoch, double error)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs<double> { Epoch = epoch, Error = error });
        }

        private void RaiseEpochEnd(int epoch, double error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs<double> { Epoch = epoch, Error = error });
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


        public ILayerSaveInfo<double> GetSaveInfo()
        {
            return new LayerSaveInfoD(NumVisibleElements, NumHiddenElements, Weights.CopyLocal());
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

        ~CudaRbmD()
        {
            Dispose(false);
        }
    }
}