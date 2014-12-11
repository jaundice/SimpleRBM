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
    public class CompareCudaVsMultidimMatrixOpsF
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
            float[,] netMatrix;
            Matrix2D<float> cudaMatrix;

            TestHelper.CreateRandomMatricesF(_dev, 10, 10, out netMatrix, out cudaMatrix);

            Assert.IsTrue(MatricesEqual(netMatrix, cudaMatrix.CopyLocal()));

            cudaMatrix.Dispose();
        }

        //failing - vital.
        [TestMethod]
        public void MatrixMultiplyEqual()
        {
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 256, 256, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 256, 256, out netMatrix2, out cudaMatrix2);

                float[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                float[,] cudaLocal = cudaM.CopyLocal();

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
        public void IdentityEqual()
        {
            Matrix2D<float> cudaMatrix1 = null;
            try
            {
                float[,] netMatrix1 = Matrix2D.IdentityF(256);

                cudaMatrix1 = _dev.AllocateAndSet<float>(256, 256);
                cudaMatrix1.Identity();
                float[,] cudaLocal = cudaMatrix1.CopyLocal();

                Assert.IsTrue(MatricesEqual(netMatrix1, cudaLocal));
            }
            finally
            {
                cudaMatrix1.Dispose();
            }
        }

        [TestMethod]
        public void TestCoverage()
        {
            Matrix2D<float> cudaMatrix1 = null;
            {
                try
                {
                    cudaMatrix1 = _dev.AllocateAndSet<float>(500, 500);
                    var netMatrix = Matrix2D.OnesF(500, 500);

                    cudaMatrix1.Increment();

                    var localCuda = cudaMatrix1.CopyLocal();
                    Assert.IsTrue(MatricesEqual(netMatrix, localCuda));

                }
                finally
                {
                    cudaMatrix1.Dispose();
                }
            }
        }

        [TestMethod]
        public void MatrixMultiplyOnesEqual()
        {
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix2, out cudaMatrix2);

                netMatrix1 = Matrix2D.OnesF(128, 128);
                netMatrix2 = Matrix2D.OnesF(128, 128);

                cudaMatrix1.Ones();
                cudaMatrix2.Ones();

                float[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix2, out cudaMatrix2);

                //Matrix2D.Fill(netMatrix1, 0.000000000000000005);
                Matrix2D.Fill(netMatrix2, 0.0003f);

                //cudaMatrix1.Fill(0.000000000000000005);
                cudaMatrix2.Fill(0.0003f);

                float[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                float[,] cudaLocal = cudaM.CopyLocal();

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
        public void MatrixMultiplyIdentityEqual()
        {
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1 = new float[36,36];
                Matrix2D.Fill(netMatrix1, 15f);
                float[,] netMatrix2 = Matrix2D.IdentityF(36);

                cudaMatrix1 = _dev.AllocateAndSet<float>(36, 36);
                cudaMatrix1.Fill(15f);

                cudaMatrix2 = _dev.AllocateAndSet<float>(36, 36);
                cudaMatrix2.Identity();

                float[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            try
            {
                float[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                float[,] localM = Matrix2D.OnesF(128, 128);

                cudaMatrix1.Ones();

                float[,] cudaLocal = cudaMatrix1.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            try
            {
                float[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                float[,] localM = Matrix2D.ZerosF(128, 128);

                cudaMatrix1.Zeros();

                float[,] cudaLocal = cudaMatrix1.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                float[,] localM = Matrix2D.Multiply(netMatrix1, 3);

                cudaM = cudaMatrix1.Multiply(3);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                float[,] localM = Matrix2D.Divide(netMatrix1, 3);

                cudaM = cudaMatrix1.Multiply(1.0f / 3);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                float[,] localM = Matrix2D.Pow(netMatrix1, 2);

                cudaM = cudaMatrix1.Pow(2);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 32, 32, out netMatrix1, out cudaMatrix1);

                float[,] localM = Matrix2D.Pow(netMatrix1, 4);

                cudaM = cudaMatrix1.Pow(4);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 32, 32, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 32, 32, out netMatrix2, out cudaMatrix2);

                float[,] localM = Matrix2D.Add(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Add(cudaMatrix2);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 32, 32, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 32, 32, out netMatrix2, out cudaMatrix2);

                float[,] localM = Matrix2D.Subtract(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Subtract(cudaMatrix2);

                float[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix1T = null;
            float[,] netMatrix1;
            try
            {
                TestHelper.CreateRandomMatricesF(_dev, 20, 10, out netMatrix1, out cudaMatrix1);
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


        public bool MatricesEqual(float[,] a, float[,] b)
        {
            float[,] difference = Matrix2D.Subtract(a, b);

            VisualizeDifferences(difference);

            return Matrix2D.EnumerateElements(difference).All(c => c < float.Epsilon);
        }


        /// <summary>
        /// displays a grid of 1s and 0s. 1 means that the element is non zero
        /// </summary>
        /// <param name="difference"></param>
        private void VisualizeDifferences(float[,] difference)
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

  
}