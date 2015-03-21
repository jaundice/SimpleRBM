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
    public class CompareCudaVsMultidimMatrixOpsF : CudaTestBase
    {
       
      
        [TestMethod]
        public void CopyToAndFromCudaEqual()
        {
            float[,] netMatrix;
            Matrix2D<float> cudaMatrix;

            TestHelper.CreateRandomMatricesF(_dev, 10, 10, out netMatrix, out cudaMatrix);

            Assert.IsTrue(MatricesEqual(netMatrix, cudaMatrix.CopyLocal()));

            cudaMatrix.Dispose();
        }

        [TestMethod]
        public void MatrixMultiplyIntEqual()
        {
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesIntF(_dev, 512, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntF(_dev, 512, 512, out netMatrix2, out cudaMatrix2);

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
        public void MatrixMultiplyIntEqual2()
        {
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesIntF(_dev, 120, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntF(_dev, 512, 120, out netMatrix2, out cudaMatrix2);

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
        public void MatrixMultiplyRealEqual()
        {
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 512, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 512, 512, out netMatrix2, out cudaMatrix2);

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
        public void MatrixMultiplyRealEqual2()
        {
            Matrix2D<float> cudaMatrix1 = null;
            Matrix2D<float> cudaMatrix2 = null;
            Matrix2D<float> cudaM = null;
            try
            {
                float[,] netMatrix1;
                float[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 120, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 512, 120, out netMatrix2, out cudaMatrix2);

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
                TestHelper.CreateRandomMatricesIntF(_dev, 100, 10000, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntF(_dev, 10000, 100, out netMatrix2, out cudaMatrix2);

                //Matrix2D.Fill(netMatrix1, 0.000000000000000005);
                var s = 3.250f;

                Matrix2D.Fill(netMatrix2, s);

                //cudaMatrix1.Fill(0.000000000000000005);
                cudaMatrix2.Fill(s);

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


        private const float Epsilon = 1E-5F;

        public bool MatricesEqual(float[,] a, float[,] b)
        {
            float[,] difference = Matrix2D.Subtract(a, b);

            var maxDiff = Matrix2D.EnumerateElements(difference).Max(v => Math.Abs(v));

            Console.WriteLine("Max Difference: {0}", maxDiff);

            VisualizeDifferences(difference);

            return Matrix2D.EnumerateElements(difference).All(c => c < Epsilon);
        }

        private void VisualizeDifferences(float[,] difference)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < difference.GetLength(0); i++)
            {
                sb.AppendLine();
                for (int j = 0; j < difference.GetLength(1); j++)
                {
                    sb.Append(Math.Abs(difference[i, j]) < Epsilon ? "0" : "1");
                }
            }
            Console.WriteLine(sb.ToString());
            
        }
    }

  
}