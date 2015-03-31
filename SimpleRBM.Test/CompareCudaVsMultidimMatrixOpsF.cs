using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using Cudafy.Translator;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Cuda;
using SimpleRBM.MultiDim;
using TElement = System.Single;
namespace SimpleRBM.Test
{
    [TestClass]
    public class CompareCudaVsMultidimMatrixOpsF : CudaTestBase
    {


        private static List<Tuple<int, int>> _matrixDimensions = new List<Tuple<int, int>>(new[]
        {
             Tuple.Create(64,64),
            Tuple.Create(64,32),
            Tuple.Create(32,64),
            Tuple.Create(64,64),
            Tuple.Create(64,64),
            Tuple.Create(64,128),
            Tuple.Create(128,64),
            Tuple.Create(64,256),
            Tuple.Create(256,64),
            Tuple.Create(64,512),
            Tuple.Create(512,64),
            Tuple.Create(64,1024),
            Tuple.Create(1024,64),
            Tuple.Create(64,2048),
            Tuple.Create(2048,64),
            Tuple.Create(64,4096),
            Tuple.Create(4096,64),
            Tuple.Create(32,32),
            Tuple.Create(32,32),
            Tuple.Create(32,32),
            Tuple.Create(32,64),
            Tuple.Create(64,32),
            Tuple.Create(32,128),
            Tuple.Create(128,32),
            Tuple.Create(32,256),
            Tuple.Create(256,32),
            Tuple.Create(32,512),
            Tuple.Create(512,32),
            Tuple.Create(32,1024),
            Tuple.Create(1024,32),
            Tuple.Create(32,2048),
            Tuple.Create(2048,32),
            Tuple.Create(32,4096),
            Tuple.Create(4096,32),
            Tuple.Create(64,64),
            Tuple.Create(64,32),
            Tuple.Create(32,64),
            Tuple.Create(64,64),
            Tuple.Create(64,64),
            Tuple.Create(64,128),
            Tuple.Create(128,64),
            Tuple.Create(64,256),
            Tuple.Create(256,64),
            Tuple.Create(64,512),
            Tuple.Create(512,64),
            Tuple.Create(64,1024),
            Tuple.Create(1024,64),
            Tuple.Create(64,2048),
            Tuple.Create(2048,64),
            Tuple.Create(64,4096),
            Tuple.Create(4096,64),
            Tuple.Create(128,128),
            Tuple.Create(128,32),
            Tuple.Create(32,128),
            Tuple.Create(128,64),
            Tuple.Create(64,128),
            Tuple.Create(128,128),
            Tuple.Create(128,128),
            Tuple.Create(128,256),
            Tuple.Create(256,128),
            Tuple.Create(128,512),
            Tuple.Create(512,128),
            Tuple.Create(128,1024),
            Tuple.Create(1024,128),
            Tuple.Create(128,2048),
            Tuple.Create(2048,128),
            Tuple.Create(128,4096),
            Tuple.Create(4096,128),
            Tuple.Create(256,256),
            Tuple.Create(256,32),
            Tuple.Create(32,256),
            Tuple.Create(256,64),
            Tuple.Create(64,256),
            Tuple.Create(256,128),
            Tuple.Create(128,256),
            Tuple.Create(256,256),
            Tuple.Create(256,256),
            Tuple.Create(256,512),
            Tuple.Create(512,256),
            Tuple.Create(256,1024),
            Tuple.Create(1024,256),
            Tuple.Create(256,2048),
            Tuple.Create(2048,256),
            Tuple.Create(256,4096),
            Tuple.Create(4096,256),
            Tuple.Create(512,512),
            Tuple.Create(512,32),
            Tuple.Create(32,512),
            Tuple.Create(512,64),
            Tuple.Create(64,512),
            Tuple.Create(512,128),
            Tuple.Create(128,512),
            Tuple.Create(512,256),
            Tuple.Create(256,512),
            Tuple.Create(512,512),
            Tuple.Create(512,512),
            Tuple.Create(512,1024),
            Tuple.Create(1024,512),
            Tuple.Create(512,2048),
            Tuple.Create(2048,512),
            Tuple.Create(512,4096),
            Tuple.Create(4096,512),
            Tuple.Create(1024,1024),
            Tuple.Create(1024,32),
            Tuple.Create(32,1024),
            Tuple.Create(1024,64),
            Tuple.Create(64,1024),
            Tuple.Create(1024,128),
            Tuple.Create(128,1024),
            Tuple.Create(1024,256),
            Tuple.Create(256,1024),
            Tuple.Create(1024,512),
            Tuple.Create(512,1024),
            Tuple.Create(1024,1024),
            Tuple.Create(1024,1024),
            Tuple.Create(1024,2048),
            Tuple.Create(2048,1024),
            Tuple.Create(1024,4096),
            Tuple.Create(4096,1024),
            Tuple.Create(2048,2048),
            Tuple.Create(2048,32),
            Tuple.Create(32,2048),
            Tuple.Create(2048,64),
            Tuple.Create(64,2048),
            Tuple.Create(2048,128),
            Tuple.Create(128,2048),
            Tuple.Create(2048,256),
            Tuple.Create(256,2048),
            Tuple.Create(2048,512),
            Tuple.Create(512,2048),
            Tuple.Create(2048,1024),
            Tuple.Create(1024,2048),
            Tuple.Create(2048,2048),
            Tuple.Create(2048,2048),
            Tuple.Create(2048,4096),
            Tuple.Create(4096,2048),
            Tuple.Create(784,500),
            Tuple.Create(500,784)
        }.Distinct().OrderBy(a => a.Item1).ThenBy(a => a.Item2));

        [TestMethod]
        public void CopyToAndFromCudaEqual()
        {
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoTestCopy(matrixDimension.Item1, matrixDimension.Item2);
            }
        }
        private void DoTestCopy(int x, int y)
        {
            TElement[,] netMatrix;
            Matrix2D<TElement> cudaMatrix;

            TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix, out cudaMatrix);

            Assert.IsTrue(MatricesEqual(netMatrix, cudaMatrix.CopyLocal()));

            cudaMatrix.Dispose();
        }

        [TestMethod]
        public void MatrixMultiplyIntEqual()
        {
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoMatrixMultiplyIntEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoMatrixMultiplyIntEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix2 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TElement[,] netMatrix2;
                TestHelper.CreateRandomMatricesIntF(_dev, x, y, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntF(_dev, y, x, out netMatrix2, out cudaMatrix2);

                TElement[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoMatrixMultiplyRealEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoMatrixMultiplyRealEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix2 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TElement[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, y, x, out netMatrix2, out cudaMatrix2);

                TElement[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions.Where(a => a.Item1 == a.Item2))
            {
                DoIdentityEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoIdentityEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            try
            {
                TElement[,] netMatrix1 = Matrix2D.IdentityF(x);

                cudaMatrix1 = _dev.AllocateAndSet<TElement>(x, x);
                cudaMatrix1.Identity();
                TElement[,] cudaLocal = cudaMatrix1.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoTestCoverage(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoTestCoverage(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            {
                try
                {
                    cudaMatrix1 = _dev.AllocateAndSet<TElement>(x, y);
                    TElement[,] netMatrix = Matrix2D.OnesF(x, y);

                    cudaMatrix1.Increment();

                    TElement[,] localCuda = cudaMatrix1.CopyLocal();
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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoMatrixMultiplyOnes(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoMatrixMultiplyOnes(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix2 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TElement[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, y, x, out netMatrix2, out cudaMatrix2);

                netMatrix1 = Matrix2D.OnesF(x, y);
                netMatrix2 = Matrix2D.OnesF(y, x);

                cudaMatrix1.Ones();
                cudaMatrix2.Ones();

                TElement[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoMatrixMultiplyFill(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoMatrixMultiplyFill(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix2 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TElement[,] netMatrix2;
                TestHelper.CreateRandomMatricesIntF(_dev, x, y, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesIntF(_dev, y, x, out netMatrix2, out cudaMatrix2);

                //Matrix2D.Fill(netMatrix1, 0.000000000000000005);
                var s = 3.250f;

                Matrix2D.Fill(netMatrix2, s);

                //cudaMatrix1.Fill(0.000000000000000005);
                cudaMatrix2.Fill(s);

                TElement[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions.Where(a => a.Item1 == a.Item2))
            {
                DoMatrixMultiplyIdentity(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoMatrixMultiplyIdentity(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix2 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1 = new TElement[x, x];
                Matrix2D.Fill(netMatrix1, 15.0f);
                TElement[,] netMatrix2 = Matrix2D.IdentityF(x);

                cudaMatrix1 = _dev.AllocateNoSet<TElement>(x, x);
                cudaMatrix1.Fill(15.0f);

                cudaMatrix2 = _dev.AllocateAndSet<TElement>(x, x);
                cudaMatrix2.Identity();

                TElement[,] localM = Matrix2D.Multiply(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Multiply(cudaMatrix2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoOnesEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoOnesEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            try
            {
                TElement[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix1, out cudaMatrix1);

                TElement[,] localM = Matrix2D.OnesF(x, y);

                cudaMatrix1.Ones();

                TElement[,] cudaLocal = cudaMatrix1.CopyLocal();

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
            Matrix2D<TElement> cudaMatrix1 = null;
            try
            {
                TElement[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                TElement[,] localM = Matrix2D.ZerosF(128, 128);

                cudaMatrix1.Zeros();

                TElement[,] cudaLocal = cudaMatrix1.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoScalarMultiplyEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoScalarMultiplyEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix1, out cudaMatrix1);

                TElement[,] localM = Matrix2D.Multiply(netMatrix1, 3);

                cudaM = cudaMatrix1.Multiply(3);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoScalarDivideEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoScalarDivideEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 2048, 2048, out netMatrix1, out cudaMatrix1);

                TElement[,] localM = Matrix2D.Divide(netMatrix1, 3);

                cudaM = cudaMatrix1.Multiply(1.0f / 3);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TestHelper.CreateRandomMatricesF(_dev, 128, 128, out netMatrix1, out cudaMatrix1);

                TElement[,] localM = Matrix2D.Pow(netMatrix1, 2);

                cudaM = cudaMatrix1.Pow(2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TestHelper.CreateRandomMatricesIntF(_dev, 32, 32, out netMatrix1, out cudaMatrix1);

                TElement[,] localM = Matrix2D.Pow(netMatrix1, 4);

                cudaM = cudaMatrix1.Pow(4);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoAddEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }


        private void DoAddEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix2 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TElement[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, 2048, 2048, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, 2048, 2048, out netMatrix2, out cudaMatrix2);

                TElement[,] localM = Matrix2D.Add(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Add(cudaMatrix2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoSubtractEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }


        private void DoSubtractEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix2 = null;
            Matrix2D<TElement> cudaM = null;
            try
            {
                TElement[,] netMatrix1;
                TElement[,] netMatrix2;
                TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix1, out cudaMatrix1);
                TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix2, out cudaMatrix2);

                TElement[,] localM = Matrix2D.Subtract(netMatrix1, netMatrix2);

                cudaM = cudaMatrix1.Subtract(cudaMatrix2);

                TElement[,] cudaLocal = cudaM.CopyLocal();

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
            foreach (var matrixDimension in _matrixDimensions)
            {
                DoTransposeEqual(matrixDimension.Item1, matrixDimension.Item2);
            }
        }

        private void DoTransposeEqual(int x, int y)
        {
            Matrix2D<TElement> cudaMatrix1 = null;
            Matrix2D<TElement> cudaMatrix1T = null;
            TElement[,] netMatrix1;
            try
            {
                TestHelper.CreateRandomMatricesF(_dev, x, y, out netMatrix1, out cudaMatrix1);
                TElement[,] netT = Matrix2D.Transpose(netMatrix1);
                cudaMatrix1T = cudaMatrix1.Transpose();
                TElement[,] cudaTLocal = cudaMatrix1T.CopyLocal();
                MatricesEqual(netT, cudaTLocal);
            }
            finally
            {
                cudaMatrix1.Dispose();
                cudaMatrix1T.Dispose();
            }
        }

        private const TElement Epsilon = 1E-5f;

        public bool MatricesEqual(TElement[,] a, TElement[,] b)
        {
            TElement[,] difference = Matrix2D.Subtract(a, b);

            var maxDiff = Matrix2D.EnumerateElements(difference).Max(v => Math.Abs(v));

            Console.WriteLine("Max Difference: {0}", maxDiff);

            VisualizeDifferences(difference);




            return Matrix2D.EnumerateElements(difference).All(c => c < Epsilon);
        }

        private void VisualizeDifferences(TElement[,] difference)
        {
            return;
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

   

  
}