//#define DEBUGCUDA
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.InteropServices;
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
        private readonly dim3 _block = new dim3(32, 32);
        private readonly dim3 _grid = new dim3(16);
        private readonly GPGPURAND _rand;
        private float[,] _weights;

        private float[,] Weights
        {
            get { return _weights; }
            set
            {
                //if (_weights != null)
                //{
                //    _gpu.Free(_weights);
                //}
                _weights = value;
            }
        }

        private GPGPU _gpu;

        public RestrictedBoltzmannMachineF(GPGPU gpu, GPGPURAND rand, dim3 grid, dim3 block, int numVisible,
            int numHidden,
            float learningRate = 0.1f)
        {

            _gpu = gpu;
            _rand = rand;


            _grid = grid;
            _block = block;


            NumHiddenElements = numHidden;
            NumVisibleElements = numVisible;
            LearningRate = learningRate;

            Console.WriteLine("Initializing {0}", LayerName);

            var weights = _gpu.AllocateAndSet<float>(numVisible + 1, numHidden + 1);
            float[,] gaussian = _gpu.AllocateAndSet<float>(numVisible, numHidden);
            float[,] multGaussian = _gpu.AllocateAndSet<float>(numVisible, numHidden);

            _gpu.Set(weights);
            _gpu.Set(gaussian);
            _gpu.Set(multGaussian);



            _gpu.Synchronize();


            GuassianDistribution(_gpu, _rand, gaussian, numVisible, numHidden);

           _gpu.Synchronize();
            _gpu.Launch(_grid, _block, Matrix2D.MultiplyScalar, gaussian, 0.1f, multGaussian);

            _gpu.Synchronize();
            _gpu.Launch(_grid, _block, Matrix2D.InsertValuesFrom, weights, 1, 1, multGaussian, 0, 0);

            Weights = weights;
            _gpu.Free(gaussian);
            _gpu.Free(multGaussian);



            Console.WriteLine("Layer Initialized");
        }

        public string LayerName
        {
            get { return string.Format("Layer {0}x{1}", NumVisibleElements, NumHiddenElements); }
        }

        public float[,] GetHiddenLayer(float[,] srcData)
        {
            int numExamples = srcData.GetLength(0);

            var tempSrcData = _gpu.Allocate<float>(srcData);
            _gpu.CopyToDevice(srcData, tempSrcData);

            float[,] data = _gpu.AllocateAndSet<float>(numExamples, srcData.GetLength(1) + 1);

            _gpu.Launch(_grid, _block, Matrix2D.InsertValuesFrom, data, 0, 1, tempSrcData, 0, 0);
            _gpu.Free(tempSrcData);

            _gpu.Launch(_grid, _block, Matrix2D.UpdateValueAlongAxis, data, 0, 1.0f, Matrix2D.FALSE);


            float[,] hiddenActivations = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
            _gpu.Launch(_grid, _block, Matrix2D.Multiply, data, Weights, hiddenActivations);

            _gpu.Free(data);


            float[,] hiddenProbs = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
            _gpu.Launch(_grid, _block, ActivationFunctions.Logistic, hiddenActivations, hiddenProbs);

            _gpu.Free(hiddenActivations);



            float[,] uniformRand = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);


            UniformDistribution(_gpu, _rand, uniformRand, numExamples, NumHiddenElements + 1);

            float[,] hsTemp = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
            _gpu.Launch(_grid, _block, Matrix2D.GreaterThan, hiddenProbs, uniformRand, hsTemp);

            _gpu.Free(hiddenProbs);
            _gpu.Free(uniformRand);

            float[,] hiddenStates = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements);
            _gpu.Launch(_grid, _block, Matrix2D.SubMatrix, hsTemp, 0, 1, 0, 0, hiddenStates);

            _gpu.Free(hsTemp);


            var localHiddenStates = new float[numExamples, NumHiddenElements];
            _gpu.CopyFromDevice(hiddenStates, localHiddenStates);

            _gpu.Free(hiddenStates);

            return localHiddenStates;
        }

        public float[,] GetVisibleLayer(float[,] srcData)
        {
            int numExamples = srcData.GetLength(0);

            float[,] data = _gpu.AllocateAndSet<float>(numExamples, srcData.GetLength(1) + 1);
            var tempSrcData = _gpu.Allocate<float>(srcData);
            _gpu.CopyToDevice(srcData, tempSrcData);


            _gpu.Launch(_grid, _block, Matrix2D.InsertValuesFrom, data, 0, 1, tempSrcData, 0, 0);
            _gpu.Free(tempSrcData);

            float[,] transposedWeights = _gpu.AllocateAndSet<float>(NumHiddenElements + 1, NumVisibleElements + 1);
            _gpu.Launch(_grid, _block, Matrix2D.Transpose, Weights, transposedWeights);

            float[,] visibleActivations = _gpu.AllocateAndSet<float>(numExamples, NumVisibleElements + 1);
            _gpu.Launch(_grid, _block, Matrix2D.Multiply, data, transposedWeights, visibleActivations);


            _gpu.Free(data);
            _gpu.Free(transposedWeights);

            float[,] visibleProbs = _gpu.AllocateAndSet<float>(numExamples, NumVisibleElements + 1);
            _gpu.Launch(_grid, _block, ActivationFunctions.Logistic, visibleActivations, visibleProbs);

            _gpu.Free(visibleActivations);



            float[,] randomDist = _gpu.AllocateAndSet<float>(numExamples, NumVisibleElements + 1);

            UniformDistribution(_gpu, _rand, randomDist, numExamples, NumVisibleElements + 1);


            float[,] visibleStatesTemp = _gpu.AllocateAndSet<float>(numExamples, NumVisibleElements + 1);
            _gpu.Launch(_grid, _block, Matrix2D.GreaterThan, visibleProbs, randomDist, visibleStatesTemp);
            _gpu.Free(visibleProbs);
            _gpu.Free(randomDist);

            float[,] visibleStates = _gpu.AllocateAndSet<float>(numExamples, NumVisibleElements);
            _gpu.Launch(_grid, _block, Matrix2D.SubMatrix, visibleStatesTemp, 0, 1, 0, 0, visibleStates);

            _gpu.Free(visibleStatesTemp);

            var localVisStates = new float[numExamples, NumVisibleElements];
            _gpu.CopyFromDevice(visibleStates, localVisStates);
            _gpu.Free(visibleStates);
            return localVisStates;
        }

        public float[,] Reconstruct(float[,] data)
        {
            float[,] hidden = GetHiddenLayer(data);
            return GetVisibleLayer(hidden);
        }

        public float[,] DayDream(int numberOfSamples)
        {

            var data = _gpu.Allocate<float>(numberOfSamples, NumVisibleElements + 1);
            _gpu.Launch(_grid, _block, Matrix2D.Ones, data);

            var uniform = _gpu.Allocate<float>(1, NumVisibleElements);
            UniformDistribution(_gpu, _rand, uniform, 1, NumVisibleElements);
            _gpu.Launch(_grid, _block, Matrix2D.InsertValuesFrom, data, 0, 1, uniform, 0, 0);

            _gpu.Free(uniform);
            _gpu.Launch(_grid, _block, Matrix2D.UpdateValueAlongAxis, data, 0, 1f, Matrix2D.TRUE);

            for (int i = 0; i < numberOfSamples; i++)
            {

                var visible = _gpu.Allocate<float>(1, NumVisibleElements + 1);

                _gpu.Launch(_grid, _block, Matrix2D.SubMatrix, data, i, 0, 1, 0, visible);

                var hiddenActivations = _gpu.Allocate<float>(1, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Multiply, visible, Weights, hiddenActivations);

                _gpu.Free(visible);
                var hiddenProbs = _gpu.Allocate<float>(1, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, ActivationFunctions.Logistic, hiddenActivations, hiddenProbs);

                _gpu.Free(hiddenActivations);

                var uniform2 = _gpu.Allocate<float>(1, NumHiddenElements + 1);
                var hiddenStates = _gpu.Allocate<float>(1, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.GreaterThan, hiddenProbs, uniform2, hiddenStates);
                _gpu.Free(hiddenProbs);
                _gpu.Free(uniform2);


                _gpu.Launch(_grid, _block, Matrix2D.UpdateValueAlongAxis, hiddenStates, 0, 1f, Matrix2D.FALSE);

                var weightsTransposed = _gpu.Allocate<float>(NumHiddenElements + 1, NumVisibleElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Transpose, Weights, weightsTransposed);
                var visibleActivations = _gpu.Allocate<float>(1, NumVisibleElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Multiply, hiddenStates, weightsTransposed, visibleActivations);
                _gpu.Free(hiddenStates);
                _gpu.Free(weightsTransposed);

                var visibleProbs = _gpu.Allocate<float>(1, NumVisibleElements + 1);
                _gpu.Launch(_grid, _block, ActivationFunctions.Logistic, visibleActivations, visibleProbs);
                _gpu.Free(visibleActivations);


                var uniform3 = _gpu.Allocate<float>(1, NumVisibleElements + 1);
                UniformDistribution(_gpu, _rand, uniform3, 1, NumVisibleElements + 1);
                var visibleStates = _gpu.Allocate<float>(1, NumVisibleElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.GreaterThan, visibleProbs, uniform3, visibleStates);

                _gpu.Free(visibleProbs);
                _gpu.Free(uniform3);

                _gpu.Launch(_grid, _block, Matrix2D.InsertValuesFromRowOrColumn, data, visibleStates, 0, Matrix2D.FALSE,
                    i, 0);
                _gpu.Free(visibleStates);
            }


            var returnVal = _gpu.Allocate<float>(numberOfSamples, NumVisibleElements);

            _gpu.Launch(_grid, _block, Matrix2D.SubMatrix, data, 0, 1, 0, 0, returnVal);
            _gpu.Free(data);
            var localReturn = new float[numberOfSamples, NumVisibleElements];
            _gpu.CopyFromDevice(returnVal, localReturn);

            _gpu.Free(returnVal);

            return localReturn;
        }

        public float Train(float[][] data)
        {
            throw new NotImplementedException();
            //return Train(Matrix2D.JaggedToMultidimesional(data));
        }

        public Task<float> AsyncTrain(float[][] data)
        {
            throw new NotImplementedException();
            //return AsyncTrain(Matrix2D.JaggedToMultidimesional(data));
        }

        public float Train(float[,] srcData)
        {
            _gpu.Synchronize();
            float error = 0f;

            int numExamples = srcData.GetLength(0);
            int numCols = srcData.GetLength(1);

            var gpu_src = _gpu.AllocateAndSet<float>(numExamples, numCols);
            _gpu.Set(gpu_src);
            _gpu.CopyToDevice(srcData, gpu_src);

            float[,] data = _gpu.AllocateAndSet<float>(numExamples, numCols + 1);
            _gpu.Set(data);

            _gpu.Launch(_grid, _block, Matrix2D.InsertValuesFrom, data, 0, 1, gpu_src, 0, 0);
            _gpu.Launch(_grid, _block, Matrix2D.UpdateValueAlongAxis, data, 0, 1.0f, Matrix2D.FALSE);


            float[,] dataTransposed = _gpu.AllocateAndSet<float>(numCols + 1, numExamples);
            _gpu.Set(dataTransposed);
            _gpu.Launch(_grid, _block, Matrix2D.Transpose, data, dataTransposed);
            var sw = new Stopwatch();
            var errors = new List<float>();

            _gpu.Synchronize();
            int i;
            for (i = 0; ; i++)
            {
                sw.Start();



                float[,] posHiddenActivations = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Multiply, data, Weights, posHiddenActivations);


                _gpu.Synchronize();
                float[,] posHiddenProbs = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, ActivationFunctions.Logistic, posHiddenActivations, posHiddenProbs);


                _gpu.Synchronize();
                _gpu.Free(posHiddenActivations);

                float[,] uniformRandom = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);

                UniformDistribution(_gpu, _rand, uniformRandom, numExamples, NumHiddenElements + 1);

              _gpu.Synchronize();
                float[,] posHiddenStates = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.GreaterThan, posHiddenProbs, uniformRandom, posHiddenStates);
                _gpu.Free(uniformRandom);


                _gpu.Synchronize();
                float[,] posAssociations = _gpu.AllocateAndSet<float>(numCols + 1, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Multiply, dataTransposed, posHiddenProbs, posAssociations);

                _gpu.Free(posHiddenProbs);

                _gpu.Synchronize();
                float[,] weightsTransposed = _gpu.AllocateAndSet<float>(NumHiddenElements + 1, NumVisibleElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Transpose, Weights, weightsTransposed);

                float[,] negVisibleActivations = _gpu.AllocateAndSet<float>(numExamples, NumVisibleElements + 1);

                _gpu.Launch(_grid, _block, Matrix2D.Multiply, posHiddenStates, weightsTransposed,
                    negVisibleActivations);

                _gpu.Free(posHiddenStates);
                _gpu.Free(weightsTransposed);



                float[,] negVisibleProbs = _gpu.AllocateAndSet<float>(numExamples, NumVisibleElements + 1);
                _gpu.Launch(_grid, _block, ActivationFunctions.Logistic, negVisibleActivations, negVisibleProbs);
                _gpu.Free(negVisibleActivations);

                _gpu.Launch(_grid, _block, Matrix2D.UpdateValueAlongAxis, negVisibleProbs, 0, 1.0f, Matrix2D.FALSE);



                float[,] negHiddenActivations = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Multiply, negVisibleProbs, Weights, negHiddenActivations);



                float[,] negHiddenProbs = _gpu.AllocateAndSet<float>(numExamples, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, ActivationFunctions.Logistic, negHiddenActivations, negHiddenProbs);
                _gpu.Free(negHiddenActivations);

                float[,] negVisibleProbsTransposed = _gpu.AllocateAndSet<float>(NumVisibleElements + 1, numExamples);
                _gpu.Launch(_grid, _block, Matrix2D.Transpose, negVisibleProbs, negVisibleProbsTransposed);

                float[,] negAssociations = _gpu.AllocateAndSet<float>(NumVisibleElements + 1, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Multiply, negVisibleProbsTransposed, negHiddenProbs, negAssociations);
                _gpu.Free(negHiddenProbs);
                _gpu.Free(negVisibleProbsTransposed);



                float[,] posAssocMinusNegAssoc = _gpu.AllocateAndSet<float>(numCols + 1, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Subtract, posAssociations, negAssociations,
                    posAssocMinusNegAssoc);

                _gpu.Free(posAssociations);
                _gpu.Free(negAssociations);

                float[,] tmult = _gpu.AllocateAndSet<float>(numCols + 1, NumHiddenElements + 1);

                _gpu.Launch(_grid, _block, Matrix2D.MultiplyScalar, posAssocMinusNegAssoc, LearningRate / numExamples,
                    tmult);

                _gpu.Free(posAssocMinusNegAssoc);

                float[,] tweight = _gpu.AllocateAndSet<float>(NumVisibleElements + 1, NumHiddenElements + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Add, Weights, tmult, tweight);

                _gpu.Free(Weights);
                _gpu.Free(tmult);

                Weights = tweight;

                float[,] delta = _gpu.AllocateAndSet<float>(numExamples, numCols + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Subtract, data, negVisibleProbs, delta);
                _gpu.Free(negVisibleProbs);

                float[,] pow = _gpu.AllocateAndSet<float>(numExamples, numCols + 1);
                _gpu.Launch(_grid, _block, Matrix2D.Pow, delta, 2.0f, pow);
                _gpu.Free(delta);

                error = Sum(_gpu, pow, numExamples);

                _gpu.Free(pow);
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

            _gpu.Free(gpu_src);

            _gpu.Free(dataTransposed);
            _gpu.Free(data);

            RaiseTrainEnd(i, error);

            return error;
        }


        public static float Sum(GPGPU gpu, float[,] matrix, int x)
        {
            var working = gpu.Allocate<float>(x, 1);
            gpu.Launch(new dim3(16), new dim3(1024), SumMatrixRows, matrix, working);

            var working2 = gpu.Allocate<float>(1, 1);
            gpu.Launch(new dim3(1), new dim3(1), SumMatrixColumns, working, working2);


            var local = new float[1, 1];
            gpu.CopyFromDevice(working2,local);

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
                var sum = 0f;
                for (var j = 0; j < matrix.GetLength(1); j++)
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
                var sum = 0f;
                for (var j = 0; j < matrix.GetLength(0); j++)
                {
                    sum += matrix[i, j];

                }
                reduced[0, i] = sum;
                i += thread.gridDim.x * thread.blockDim.x;
            }
        }

        public Task<float> AsyncTrain(float[,] data)
        {
            return Task.Run(() => Train(data));
        }

        public event EventHandler<EpochEventArgs<float>> EpochEnd;

        public event EventHandler<EpochEventArgs<float>> TrainEnd;

        public static void GuassianDistribution(GPGPU gpu, GPGPURAND rand, float[,] array, int x, int y)
        {


            float[] tempGaussian = gpu.AllocateAndSet<float>(y);

            for (int i = 0; i < x; i++)
            {
                if (rand != null)
                    rand.GenerateNormal(tempGaussian, 0f, 1f, y);
                gpu.Launch(new dim3(1), new dim3(1024), CopyToArrayAtN, array, tempGaussian, i);
            }
            gpu.Free(tempGaussian);
        }

        
        public static void UniformDistribution(GPGPU gpu, GPGPURAND rand, float[,] array, int x, int y)
        {

            var tempUniform = gpu.AllocateAndSet<float>(y);

            for (int i = 0; i < x; i++)
            {
                rand.GenerateUniform(tempUniform, y);

                gpu.Launch(new dim3(16), new dim3(1024), CopyToArrayAtN, array, tempUniform, i);
            }
            gpu.Free(tempUniform);
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


    public static class GPGPUEx
    {
        public static T[] AllocateAndSet<T>(this GPGPU gpu, int x) where T : struct
        {
            var res = gpu.Allocate<T>(x);
            gpu.Set(res);
            return res;
        }

        public static T[,] AllocateAndSet<T>(this GPGPU gpu, int x, int y) where T : struct
        {
            var res = gpu.Allocate<T>(x, y);
            gpu.Set(res);
            return res;
        }
    }
}