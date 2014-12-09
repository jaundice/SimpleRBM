using System;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Cuda;
using SimpleRBM.MultiDim;

namespace SimpleRBM.Test
{
    [TestClass]
    public class CompareCudaVsMultidimMatrixOps
    {
        private static GPGPU _dev;
        private static GPGPURAND _rand;

        [ClassInitialize]
        public static void InitCudaDevice(TestContext context)
        {
            CudafyHost.ClearAllDeviceMemories();
            CudafyHost.ClearDevices();


            _dev = CudafyHost.GetDevice(eGPUType.Cuda, 0);

            GPGPUProperties props = _dev.GetDeviceProperties();
            Console.WriteLine(props.Name);

            Console.WriteLine("Compiling CUDA module");

            eArchitecture arch = _dev.GetArchitecture();
            ePlatform plat = Environment.Is64BitProcess ? ePlatform.x64 : ePlatform.x86;

            if (plat == ePlatform.x64)
                throw new Exception("CUDA Random will fail currently on x64");

            CudafyModule mod = CudafyTranslator.Cudafy(
                plat,
                arch,
                typeof(ActivationFunctionsCuda),
                typeof(Matrix2DCudaF),
                typeof(Matrix2DCudaD),
                typeof(CudaRbmF),
                typeof(CudaRbmD)
                );


            ThreadOptimiser.Instance = new ThreadOptimiser(props.Capability, props.MultiProcessorCount,
                props.MaxThreadsPerBlock,
                props.MaxThreadsPerMultiProcessor, props.MaxGridSize, props.MaxThreadsSize);

            _rand = GPGPURAND.Create(_dev, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

            _rand.SetPseudoRandomGeneratorSeed((ulong)DateTime.Now.Ticks);
            _rand.GenerateSeeds();

            Console.WriteLine("Loading Module");
            _dev.LoadModule(mod);
        }

        [TestMethod]
        public void CopyToAndFromCudaEqual()
        {
            double[,] netMatrix;
            Matrix2D<double> cudaMatrix;

            TestHelper.CreateRandomMatrices(_dev, 10, 10, out netMatrix, out cudaMatrix);

            Assert.IsTrue(MatricesEqual(netMatrix, cudaMatrix.CopyLocal()));

            cudaMatrix.Dispose();
        }

        //failing - vital.
        [TestMethod]
        public void MatrixMultiplyEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatrices(_dev, 128, 64, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatrices(_dev, 64, 128, out netMatrix2, out cudaMatrix2);

                double[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix2.Dispose();
                cudaM.Dispose();
            }
        }

        [TestMethod]
        public void MatrixMultiplyOnesEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix2, out cudaMatrix2);

                netMatrix1 = Matrix2D.OnesD(128, 128);
                netMatrix2 = Matrix2D.OnesD(128, 128);

                cudaMatrix1.Ones();
                cudaMatrix2.Ones();

                double[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix2.Dispose();
                cudaM.Dispose();
            }
        }


        [TestMethod]
        public void MatrixMultiplyFillEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix2, out cudaMatrix2);

                //Matrix2D.Fill(netMatrix1, 0.000000000000000005);
                Matrix2D.Fill(netMatrix2, 0.0003);

                //cudaMatrix1.Fill(0.000000000000000005);
                cudaMatrix2.Fill(0.0003);

                double[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix2.Dispose();
                cudaM.Dispose();
            }
        }

        [TestMethod]
        public void OnesEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            try
            {
                double[,] netMatrix1;
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                double[,] localM = Matrix2D.OnesD(128, 128);

                cudaMatrix1.Ones();

                double[,] cudaLocal = cudaMatrix1.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
            }
        }

        [TestMethod]
        public void ZerosEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            try
            {
                double[,] netMatrix1;
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                double[,] localM = Matrix2D.ZerosD(128, 128);

                cudaMatrix1.Zeros();

                double[,] cudaLocal = cudaMatrix1.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
            }
        }

        [TestMethod]
        public void ScalarMultiplyEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                double[,] localM = Matrix2D.Multiply(netMatrix1, 3);

                cudaM = cudaMatrix1.Multiply(3);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaM.Dispose();
            }
        }

        [TestMethod]
        public void ScalarDivideEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                double[,] localM = Matrix2D.Divide(netMatrix1, 3);

                cudaM = cudaMatrix1.Multiply(1.0 / 3);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaM.Dispose();
            }
        }

        [TestMethod]
        public void Pow2Equal()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                TestHelper.CreateRandomMatrices(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                double[,] localM = Matrix2D.Pow(netMatrix1, 2);

                cudaM = cudaMatrix1.Pow(2);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaM.Dispose();
            }
        }
        //failing but only pow2 is used in the app so far
        [TestMethod]
        public void Pow4Equal()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                TestHelper.CreateRandomMatrices(_dev, 32, 32, out netMatrix1, out cudaMatrix1);

                double[,] localM = Matrix2D.Pow(netMatrix1, 4);

                cudaM = cudaMatrix1.Pow(4);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaM.Dispose();
            }
        }

        [TestMethod]
        public void AddEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatrices(_dev, 32, 32, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatrices(_dev, 32, 32, out netMatrix2, out cudaMatrix2);

                double[,] localM = Matrix2D.Add(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Add(cudaMatrix2);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix2.Dispose();
                cudaM.Dispose();
            }
        }


        [TestMethod]
        public void SubtractEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatrices(_dev, 32, 32, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatrices(_dev, 32, 32, out netMatrix2, out cudaMatrix2);

                double[,] localM = Matrix2D.Subtract(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Subtract(cudaMatrix2);

                double[,] cudaLocal = cudaM.CopyLocal();

                Assert.IsTrue(MatricesEqual(localM, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix2.Dispose();
                cudaM.Dispose();
            }
        }




        [TestMethod]
        public void TransposeEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix1T = null;
            double[,] netMatrix1;
            try
            {
                TestHelper.CreateRandomMatrices(_dev, 20, 10, out netMatrix1, out cudaMatrix1);
                var netT = Matrix2D.Transpose(netMatrix1);
                cudaMatrix1T = cudaMatrix1.Transpose();
                var cudaTLocal = cudaMatrix1T.CopyLocal();
                MatricesEqual(netT, cudaTLocal);

            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix1T.Dispose();
            }
        }

        public bool MatricesEqual(double[,] a, double[,] b)
        {
            double[,] difference = Matrix2D.Subtract(a, b);

            VisualizeDifferences(difference);

            return Matrix2D.EnumerateElements(difference).All(c => c < float.Epsilon);
        }

        private void VisualizeDifferences(double[,] difference)
        {
            StringBuilder sb = new StringBuilder();
            for (var i = 0; i < difference.GetLength(0); i++)
            {
                sb.AppendLine();
                for (var j = 0; j < difference.GetLength(1); j++)
                {
                    sb.Append(Math.Abs(difference[i, j]) < float.Epsilon ? "0" : "1");
                }
            }
            Console.Write(sb.ToString());
        }


        [ClassCleanup]
        public static void Cleanup()
        {
            _rand.Dispose();
            _dev.Dispose();
        }
    }

    public static class TestHelper
    {
        public static void CreateRandomMatrices(GPGPU dev, int rows, int cols, out double[,] netMatrix,
            out Matrix2D<double> cudaMatrix)
        {
            netMatrix = Distributions.GaussianMatrix(rows, cols);

            cudaMatrix = dev.AllocateAndSet<double>(rows, cols);

            dev.CopyToDevice(netMatrix, cudaMatrix);
        }
    }
}