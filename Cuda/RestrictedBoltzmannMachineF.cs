//#define DEBUGCUDA

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;

namespace CudaRbm
{
    public class RestrictedBoltzmannMachineF : IRestrictedBoltzmannMachine<float>
    {
        private readonly float LearningRate;
        private readonly int NumHiddenElements;
        public readonly int NumVisibleElements;
        private readonly GPGPU _gpu;
        private readonly GPGPURAND _rand;

        public RestrictedBoltzmannMachineF(GPGPU gpu, GPGPURAND rand, int numVisible,
            int numHidden,
            float learningRate = 0.1f)
        {
            _gpu = gpu;
            _rand = rand;



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

        private Matrix2D<float> Weights { get; set; }

        public string LayerName
        {
            get { return string.Format("Layer {0}x{1}", NumVisibleElements, NumHiddenElements); }
        }

        public float[,] GetHiddenLayer(float[,] srcData)
        {
            int numExamples = srcData.GetLength(0);

            Matrix2D<float> tempSrcData = _gpu.AllocateAndSet<float>(srcData.GetLength(0), srcData.GetLength(1));
            _gpu.CopyToDevice(srcData, tempSrcData);

            Matrix2D<float> data = _gpu.AllocateAndSet<float>(numExamples, srcData.GetLength(1) + 1);

            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            data.UpdateValuesAlongAxis(0, 1.0f, Axis.Column);

            var hiddenActivations = data.Multiply(Weights);


            data.Dispose();

            var hiddenProbs = hiddenActivations.Logistic();

            hiddenActivations.Dispose();


            Matrix2D<float> uniformRand = UniformDistribution(_gpu, _rand, numExamples, NumHiddenElements + 1);

            var hsTemp = hiddenProbs.GreaterThan(uniformRand);

            hiddenProbs.Dispose();
            uniformRand.Dispose();

            var hiddenStates = hsTemp.SubMatrix(0, 1);

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
            var tempSrcData = MatrixEx.Upload(_gpu, srcData);


            data.InsertValuesFrom(0, 1, tempSrcData);

            tempSrcData.Dispose();

            var transposedWeights = Weights.Transpose();

            var visibleActivations = data.Multiply(transposedWeights);


            data.Dispose();
            transposedWeights.Dispose();

            var visibleProbs = visibleActivations.Logistic();

            visibleActivations.Dispose();


            Matrix2D<float> randomDist = UniformDistribution(_gpu, _rand, numExamples, NumVisibleElements + 1);

            var visibleStatesTemp = visibleProbs.GreaterThan(randomDist);

            visibleProbs.Dispose();
            randomDist.Dispose();

            var visibleStates = visibleStatesTemp.SubMatrix(0, 1);


            visibleStatesTemp.Dispose();

            var localVisStates = visibleStates.CopyLocal();

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
            //_gpu.Launch(_grid, _block, Matrix2DCuda.Ones, data.Matrix);
            data.Ones();

            Matrix2D<float> uniform = UniformDistribution(_gpu, _rand, 1, NumVisibleElements);

            //_gpu.Launch(_grid, _block, Matrix2DCuda.InsertValuesFrom, data.Matrix, 0, 1, uniform.Matrix, 0, 0);

            data.InsertValuesFrom(0, 1, uniform);

            uniform.Dispose();


            //_gpu.Launch(_grid, _block, Matrix2DCuda.UpdateValueAlongAxis, data.Matrix, 0, 1f, Matrix2DCuda.TRUE);

            data.UpdateValuesAlongAxis(0, 1f, Axis.Row);


            for (int i = 0; i < numberOfSamples; i++)
            {
                //Matrix2D<float> visible = _gpu.AllocateAndSet<float>(1, NumVisibleElements + 1);
                //_gpu.Launch(_grid, _block, Matrix2DCuda.SubMatrix, data.Matrix, i, 0, 1, 0, visible.Matrix);

                var visible = data.SubMatrix(i, 0, 1, 0);


                //Matrix2D<float> hiddenActivations = _gpu.AllocateAndSet<float>(1, NumHiddenElements + 1);
                //_gpu.Launch(_grid, _block, Matrix2DCuda.Multiply, visible.Matrix, Weights.Matrix, hiddenActivations.Matrix);

                var hiddenActivations = visible.Multiply(Weights);

                visible.Dispose();


                //Matrix2D<float> hiddenProbs = _gpu.AllocateAndSet<float>(1, NumHiddenElements + 1);
                //_gpu.Launch(_grid, _block, ActivationFunctionsCuda.Logistic, hiddenActivations.Matrix, hiddenProbs.Matrix);

                var hiddenProbs = hiddenActivations.Logistic();

                hiddenActivations.Dispose();

                Matrix2D<float> uniform2 = UniformDistribution(_gpu, _rand, 1, NumHiddenElements + 1);
                //Matrix2D<float> hiddenStates = _gpu.AllocateAndSet<float>(1, NumHiddenElements + 1);
                //_gpu.Launch(_grid, _block, Matrix2DCuda.GreaterThan, hiddenProbs.Matrix, uniform2.Matrix,
                //    hiddenStates.Matrix);
                var hiddenStates = hiddenProbs.GreaterThan(uniform2);

                hiddenProbs.Dispose();
                uniform2.Dispose();


                //_gpu.Launch(_grid, _block, Matrix2DCuda.UpdateValueAlongAxis, hiddenStates.Matrix, 0, 1f, Matrix2DCuda.FALSE);

                hiddenStates.UpdateValuesAlongAxis(0, 0f, Axis.Column);



                //Matrix2D<float> weightsTransposed = _gpu.AllocateAndSet<float>(NumHiddenElements + 1,
                //    NumVisibleElements + 1);
                //_gpu.Launch(_grid, _block, Matrix2DCuda.Transpose, Weights.Matrix, weightsTransposed.Matrix);
                var weightsTransposed = Weights.Transpose();



                //Matrix2D<float> visibleActivations = _gpu.AllocateAndSet<float>(1, NumVisibleElements + 1);
                //_gpu.Launch(_grid, _block, Matrix2DCuda.Multiply, hiddenStates.Matrix, weightsTransposed.Matrix,
                //    visibleActivations.Matrix);

                var visibleActivations = hiddenStates.Multiply(weightsTransposed);

                hiddenStates.Dispose();
                weightsTransposed.Dispose();

                //Matrix2D<float> visibleProbs = _gpu.AllocateAndSet<float>(1, NumVisibleElements + 1);
                //_gpu.Launch(_grid, _block, ActivationFunctionsCuda.Logistic, visibleActivations.Matrix, visibleProbs.Matrix);

                var visibleProbs = visibleActivations.Logistic();

                visibleActivations.Dispose();


                Matrix2D<float> uniform3 = UniformDistribution(_gpu, _rand, 1, NumVisibleElements + 1);

                //Matrix2D<float> visibleStates = _gpu.AllocateAndSet<float>(1, NumVisibleElements + 1);
                //_gpu.Launch(_grid, _block, Matrix2DCuda.GreaterThan, visibleProbs.Matrix, uniform3.Matrix,
                //    visibleStates.Matrix);

                var visibleStates = visibleProbs.GreaterThan(uniform3);

                visibleProbs.Dispose();
                uniform3.Dispose();

                //_gpu.Launch(_grid, _block, Matrix2DCuda.InsertValuesFromRowOrColumn, data.Matrix, visibleStates.Matrix, 0,
                //    Matrix2DCuda.FALSE,
                //    i, 0);

                data.InsertValuesFromRowOrColumn(visibleStates, 0, Axis.Row, i, 0);


                visibleStates.Dispose();
            }


            //Matrix2D<float> returnVal = _gpu.AllocateAndSet<float>(numberOfSamples, NumVisibleElements);

            //_gpu.Launch(_grid, _block, Matrix2DCuda.SubMatrix, data.Matrix, 0, 1, 0, 0, returnVal.Matrix);

            var returnVal = data.SubMatrix(0, 1);
            data.Dispose();
            //var localReturn = new float[numberOfSamples, NumVisibleElements];
            //_gpu.CopyFromDevice(returnVal, localReturn);

            var localReturn = returnVal.CopyLocal();

            returnVal.Dispose();

            return localReturn;
        }

        public float Train(float[][] data)
        {
            return Train(Matrix2DCuda.JaggedToMultidimesional(data));
        }

        public Task<float> AsyncTrain(float[][] data)
        {
            return AsyncTrain(Matrix2DCuda.JaggedToMultidimesional(data));
        }

        public float Train(float[,] srcData)
        {
            _gpu.Synchronize();
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

                        var posHiddenActivations = data.Multiply(Weights);

                        var posHiddenProbs = posHiddenActivations.Logistic();


                        posHiddenActivations.Dispose();

                        Matrix2D<float> uniformRandom = UniformDistribution(_gpu, _rand, numExamples, NumHiddenElements + 1);

                        var posHiddenStates = posHiddenProbs.GreaterThan(uniformRandom);

                        uniformRandom.Dispose();

                        var posAssociations = dataTransposed.Multiply(posHiddenProbs);

                        posHiddenProbs.Dispose();

                        var weightsTransposed = Weights.Transpose();

                        var negVisibleActivations = posHiddenStates.Multiply(weightsTransposed);

                        posHiddenStates.Dispose();
                        weightsTransposed.Dispose();

                        var negVisibleProbs = negVisibleActivations.Logistic();


                        negVisibleActivations.Dispose();

                        negVisibleProbs.UpdateValuesAlongAxis(0, 1f, Axis.Column);

                        var negHiddenActivations = negVisibleProbs.Multiply(Weights);

                        var negHiddenProbs = negHiddenActivations.Logistic();

                        negHiddenActivations.Dispose();

                        var negVisibleProbsTransposed = negVisibleProbs.Transpose();

                        var negAssociations = negVisibleProbsTransposed.Multiply(negHiddenProbs);

                        negHiddenProbs.Dispose();
                        negVisibleProbsTransposed.Dispose();

                        var posAssocMinusNegAssoc = posAssociations.Subtract(negAssociations);

                        posAssociations.Dispose();
                        negAssociations.Dispose();

                        var tmult = posAssocMinusNegAssoc.Multiply(LearningRate / numExamples);

                        posAssocMinusNegAssoc.Dispose();

                        var tweight = Weights.Add(tmult);

                        Weights.Dispose();
                        tmult.Dispose();

                        Weights = tweight;

                        var delta = data.Subtract(negVisibleProbs);


                        negVisibleProbs.Dispose();

                        var pow = delta.Pow(2.0f);

                        delta.Dispose();

                        error = Sum(_gpu, pow, numExamples);

                        pow.Dispose();
                        errors.Add(error);
                        RaiseEpochEnd(i, error);

                        if (i % 20 == 0)
                            Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error,
                                sw.ElapsedMilliseconds);
                        sw.Reset();


                        if (i > 150
                            && errors[i] > errors[i - 1]
                            && errors.Skip(Math.Max(0, i - 10)).Take(10).Average() >
                            errors.Skip(Math.Max(0, i - 150)).Take(150).Average())
                        {
                            Console.WriteLine("Error rates are increasing. Stop training");
                            break;
                        }
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
            ThreadOptimiser.Instance.GetStrategy(x, 0, out grid, out block);

            float[,] working = gpu.Allocate<float>(x, 1);
            gpu.Launch(grid, block, SumMatrixRows, matrix.Matrix, working);

            float[,] working2 = gpu.Allocate<float>(1, 1);
            gpu.Launch(new dim3(1), new dim3(1), SumMatrixColumns, working, working2);


            var local = new float[1, 1];
            gpu.CopyFromDevice(working2, local);

            gpu.Free(working);
            gpu.Free(working2);
            return local[0, 0];
        }

        [Cudafy]
        public static void SumMatrixRows(GThread thread, float[,] matrix, float[,] reduced)
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
        public static void SumMatrixColumns(GThread thread, float[,] matrix, float[,] reduced)
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
            ThreadOptimiser.Instance.GetStrategy(y, 0, out grid, out block);
            using (Matrix1D<float> tempGaussian = gpu.AllocateAndSet<float>(y))
            {
                for (int i = 0; i < x; i++)
                {
                    if (rand != null)
                        rand.GenerateNormal(tempGaussian, 0f, 1f, y);
                    gpu.Launch(grid, block, CopyToArrayAtN, array.Matrix, tempGaussian.Matrix, i);
                }
            }
            return array;
        }


        public static Matrix2D<float> UniformDistribution(GPGPU gpu, GPGPURAND rand, int x, int y)
        {
            Matrix2D<float> array = gpu.AllocateAndSet<float>(x, y);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(y, 0, out grid, out block);
            using (Matrix1D<float> tempUniform = gpu.AllocateAndSet<float>(y))
            {
                for (int i = 0; i < x; i++)
                {
                    rand.GenerateUniform(tempUniform, y);

                    gpu.Launch(grid, block, CopyToArrayAtN, array.Matrix, tempUniform.Matrix, i);
                }
            }
            return array;
        }


        [Cudafy]
        public static void CopyToArrayAtN(GThread thread, float[,] target, float[] source, int x)
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
    }
}