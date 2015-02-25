using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Cuda;

namespace SimpleRBM.Test
{
    [TestClass]
    public class UnitTest3 : CudaTestBase
    {
        [TestMethod]
        public void RepMat()
        {
            float[,] source = RandomMatrix(1, 4);

            float[,] res;

            using (var data = _dev.Upload(source))
            using (var repl = data.RepMatRows(4))
            {
                res = repl.CopyLocal();
            }

            PrintArray(res);
        }


        [TestMethod]
        public void TestSumCols()
        {
            float[,] src = new float[2, 2];
            src[0, 0] = 1f;
            src[0, 1] = 1f;
            src[1, 0] = 1f;
            src[1, 1] = 1f;
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumColumns())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }


        [TestMethod]
        public void TestSumRows()
        {
            float[,] src = new float[2, 2];
            src[0, 0] = 1f;
            src[0, 1] = 1f;
            src[1, 0] = 1f;
            src[1, 1] = 1f;
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumRows())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }


        [TestMethod]
        public void TestSumMatrix()
        {
            float[,] src = new float[2, 2];
            src[0, 0] = 1f;
            src[0, 1] = 1f;
            src[1, 0] = 1f;
            src[1, 1] = 1f;
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var cols = data.SumColumns())
            using (var res = cols.SumRows())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }


        [TestMethod]
        public void TestSumColsBig()
        {
            float[,] src = RandomMatrix(500, 1000);
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumColumns())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }

        [TestMethod]
        public void TestSumColsBig2()
        {
            float[,] src = RandomMatrix(1000, 500);
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumColumns())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }


        [TestMethod]
        public void TestSumRowsBig()
        {
            float[,] src = RandomMatrix(500, 1000);
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumRows())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }

        [TestMethod]
        public void TestSumRowsBig2()
        {
            float[,] src = RandomMatrix(1000, 500);
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumRows())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }


        [TestMethod]
        public void TestSumMatrixBig()
        {
            float[,] src = RandomMatrix(1000, 500);
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var cols = data.SumColumns())
            using (var res = cols.SumRows())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }


        [TestMethod]
        public void TestSumMatrixBig2()
        {
            float[,] src = RandomMatrix(500, 1000);
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var cols = data.SumColumns())
            using (var res = cols.SumRows())
            {
                ret = res.CopyLocal();
            }

            PrintArray(ret);
        }

        private float[,] RandomMatrix(int rows, int cols)
        {
            var m = new float[rows, cols];
            Random rnd = new Random();

            Parallel.For(0, rows, a => Parallel.For(0, cols, b =>
            {
                lock (rnd)
                    m[a, b] = (float) rnd.NextDouble();
            }));
            return m;
        }

        private void PrintArray<T>(T[,] ret)
        {
            for (int i = 0; i < ret.GetLength(0); i++)
            {
                Console.WriteLine();
                for (int j = 0; j < ret.GetLength(1); j++)
                {
                    Console.Write("{0:F2}\t", ret[i, j]);
                }
            }
        }
    }
}