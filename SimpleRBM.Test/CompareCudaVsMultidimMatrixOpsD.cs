using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy.Host;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Cuda;
using SimpleRBM.MultiDim;

namespace SimpleRBM.Test
{
    [TestClass]
    public class CompareCudaVsMultidimMatrixOpsD : CudaTestBase
    {

        [TestMethod]
        public void CopyToAndFromCudaEqual()
        {
            double[,] netMatrix;
            Matrix2D<double> cudaMatrix;

            TestHelper.CreateRandomMatricesD(_dev, 10, 10, out netMatrix, out cudaMatrix);

            Assert.IsTrue(MatricesEqual(netMatrix, cudaMatrix.CopyLocal()));

            cudaMatrix.Dispose();
        }

        [TestMethod]
        public void MatrixMultiplyIntEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatricesIntD(_dev, 512, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntD(_dev, 512, 512, out netMatrix2, out cudaMatrix2);

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
        public void MatrixMultiplyIntEqual2()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatricesIntD(_dev, 120, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntD(_dev, 512, 120, out netMatrix2, out cudaMatrix2);

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
        public void MatrixMultiplyRealEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatricesD(_dev, 512, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesD(_dev, 512, 512, out netMatrix2, out cudaMatrix2);

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
        public void MatrixMultiplyRealEqual2()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatricesD(_dev, 120, 512, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesD(_dev, 512, 120, out netMatrix2, out cudaMatrix2);

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
        public void IdentityEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            try
            {
                double[,] netMatrix1 = Matrix2D.IdentityD(256);

                cudaMatrix1 = _dev.AllocateAndSet<double>(256, 256);
                cudaMatrix1.Identity();
                double[,] cudaLocal = cudaMatrix1.CopyLocal();

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
            Matrix2D<double> cudaMatrix1 = null;
            {
                try
                {
                    cudaMatrix1 = _dev.AllocateAndSet<double>(500, 500);
                    double[,] netMatrix = Matrix2D.OnesD(500, 500);

                    cudaMatrix1.Increment();

                    double[,] localCuda = cudaMatrix1.CopyLocal();
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
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1;
                double[,] netMatrix2;
                TestHelper.CreateRandomMatricesD(_dev, 128, 128, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesD(_dev, 128, 128, out netMatrix2, out cudaMatrix2);

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
                TestHelper.CreateRandomMatricesIntD(_dev, 10, 100, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntD(_dev, 100, 10, out netMatrix2, out cudaMatrix2);

                //Matrix2D.Fill(netMatrix1, 0.000000000000000005);
                var s = 3.250;

                Matrix2D.Fill(netMatrix2, s);

                //cudaMatrix1.Fill(0.000000000000000005);
                cudaMatrix2.Fill(s);

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
        public void MatrixMultiplyIdentityEqual()
        {
            Matrix2D<double> cudaMatrix1 = null;
            Matrix2D<double> cudaMatrix2 = null;
            Matrix2D<double> cudaM = null;
            try
            {
                double[,] netMatrix1 = new double[36, 36];
                Matrix2D.Fill(netMatrix1, 15f);
                double[,] netMatrix2 = Matrix2D.IdentityD(36);

                cudaMatrix1 = _dev.AllocateAndSet<double>(36, 36);
                cudaMatrix1.Fill(15f);

                cudaMatrix2 = _dev.AllocateAndSet<double>(36, 36);
                cudaMatrix2.Identity();

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
                TestHelper.CreateRandomMatricesD(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

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
                TestHelper.CreateRandomMatricesD(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

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
                TestHelper.CreateRandomMatricesD(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

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
                TestHelper.CreateRandomMatricesD(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

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
                TestHelper.CreateRandomMatricesD(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

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
                TestHelper.CreateRandomMatricesIntD(_dev, 32, 32, out netMatrix1, out cudaMatrix1);

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
                TestHelper.CreateRandomMatricesD(_dev, 32, 32, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesD(_dev, 32, 32, out netMatrix2, out cudaMatrix2);

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
                TestHelper.CreateRandomMatricesD(_dev, 32, 32, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesD(_dev, 32, 32, out netMatrix2, out cudaMatrix2);

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
                TestHelper.CreateRandomMatricesD(_dev, 20, 10, out netMatrix1, out cudaMatrix1);
                double[,] netT = Matrix2D.Transpose(netMatrix1);
                cudaMatrix1T = cudaMatrix1.Transpose();
                double[,] cudaTLocal = cudaMatrix1T.CopyLocal();
                MatricesEqual(netT, cudaTLocal);
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix1T.Dispose();
            }
        }

        private const double Epsilon = 1E-13;

        public bool MatricesEqual(double[,] a, double[,] b)
        {
            double[,] difference = Matrix2D.Subtract(a, b);

            var maxDiff = Matrix2D.EnumerateElements(difference).Max(v => Math.Abs(v));

            Console.WriteLine("Max Difference: {0}", maxDiff);

            VisualizeDifferences(difference);




            return Matrix2D.EnumerateElements(difference).All(c => c < Epsilon);
        }

        private void VisualizeDifferences(double[,] difference)
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
            Console.Write(sb.ToString());
        }


        //[ClassCleanup]
        //public static void Cleanup()
        //{
        //    _rand.Dispose();
        //    _dev.Dispose();
        //}
    }

    public static class TestHelper
    {
        public static void CreateRandomMatricesD(GPGPU dev, int rows, int cols, out double[,] netMatrix,
            out Matrix2D<double> cudaMatrix)
        {
            netMatrix = Distributions.GaussianMatrixD(rows, cols);

            cudaMatrix = dev.AllocateAndSet<double>(rows, cols);

            dev.CopyToDevice(netMatrix, cudaMatrix);
        }

        public static void CreateRandomMatricesF(GPGPU dev, int rows, int cols, out float[,] netMatrix,
            out Matrix2D<float> cudaMatrix)
        {
            netMatrix = Distributions.GaussianMatrixF(rows, cols);

            cudaMatrix = dev.AllocateAndSet<float>(rows, cols);

            dev.CopyToDevice(netMatrix, cudaMatrix);
        }

        public static void CreateRandomMatricesIntD(GPGPU dev, int rows, int cols, out double[,] netMatrix,
           out Matrix2D<double> cudaMatrix)
        {
            var m = Distributions.GaussianMatrixD(rows, cols);
            m = Matrix2D.Multiply(m, 100);

            for (int i = 0; i < m.GetLength(0); i++)
            {
                var x = i;
                Parallel.For(0, m.GetLength(1), j =>
                {
                    m[x, j] = Math.Ceiling(m[x, j]);
                });
            }
            netMatrix = m;
            cudaMatrix = dev.AllocateAndSet<double>(rows, cols);

            dev.CopyToDevice(netMatrix, cudaMatrix);
        }

        public static void CreateRandomMatricesIntF(GPGPU dev, int rows, int cols, out float[,] netMatrix,
            out Matrix2D<float> cudaMatrix)
        {
            var m = Distributions.GaussianMatrixF(rows, cols);
            m = Matrix2D.Multiply(m, 100);

            for (int i = 0; i < m.GetLength(0); i++)
            {
                var x = i;
                Parallel.For(0, m.GetLength(1), j =>
                {
                    m[x, j] = (float)Math.Ceiling(m[x, j]);
                });
            }
            netMatrix = m;
            cudaMatrix = dev.AllocateAndSet<float>(rows, cols);

            dev.CopyToDevice(netMatrix, cudaMatrix);
        }
    }
}