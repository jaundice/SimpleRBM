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
        public void RepMatRows()
        {
            double[,] source = new double[1, 7] { { 1, 2, 3, 4, 5, 6, 7 } };

            double[,] res;

            using (var data = _dev.Upload(source))
            using (var repl = data.RepMatRows(4))
            {
                res = repl.CopyLocal();
            }
            PrintArray(source);
            Console.WriteLine();

            PrintArray(res);
        }

        [TestMethod]
        public void RepMatCols()
        {
            double[,] source = RandomMatrix(8, 1);

            double[,] res;

            using (var data = _dev.Upload(source))
            using (var repl = data.RepMatCols(5))
            {
                res = repl.CopyLocal();
            }
            PrintArray(source);
            Console.WriteLine();


            PrintArray(res);
        }


        [TestMethod]
        public void TestSumCols()
        {
            float[,] src = new float[2, 3];
            src[0, 0] = 1f;
            src[0, 1] = 2f;
            src[0, 2] = 3f;
            src[1, 0] = 1f;
            src[1, 1] = 2f;
            src[1, 2] = 3f;
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumColumns())
            {
                ret = res.CopyLocal();
            }
            PrintArray(src);
            Console.WriteLine();
            PrintArray(ret);
        }


        [TestMethod]
        public void TestSumRows()
        {
            float[,] src = new float[3, 2];
            src[0, 0] = 1f;
            src[0, 1] = 1f;
            src[1, 0] = 2f;
            src[1, 1] = 2f;
            src[2, 0] = 2f;
            src[2, 1] = 2f;
            float[,] ret;
            using (var data = _dev.Upload(src))
            using (var res = data.SumRows())
            {
                ret = res.CopyLocal();
            }
            PrintArray(src);
            Console.WriteLine();
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
            double[,] src = RandomMatrix(500, 1000);
            double[,] ret;
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
            double[,] src = RandomMatrix(1000, 500);
            double[,] ret;
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
            double[,] src = RandomMatrix(500, 1000);
            double[,] ret;
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
            double[,] src = RandomMatrix(1000, 500);
            double[,] ret;
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
            double[,] src = RandomMatrix(1000, 500);
            using (var data = _dev.Upload(src))
            {
                Console.WriteLine(data.Sum());
            }
        }


        [TestMethod]
        public void TestSumMatrixBig2()
        {
            double[,] src = RandomMatrix(500, 1000);
            using (var data = _dev.Upload(src))
            {
                Console.WriteLine(data.Sum());
            }

        }

        [TestMethod]
        public void TestTranspose()
        {
            using (var identity = _dev.GuassianDistribution(_rand, 32, 32, 0.0))
            {
                identity.ToBinary();

                using (
                    var trans = identity.Transpose())
                {
                    PrintArray(identity.CopyLocal());
                    Console.WriteLine();
                    PrintArray(trans.CopyLocal());
                }
            }


        }

        private double[,] RandomMatrix(int rows, int cols)
        {
            var m = new double[rows, cols];
            Random rnd = new Random();

            Parallel.For(0, rows, a => Parallel.For(0, cols, b =>
            {
                lock (rnd)
                    m[a, b] = rnd.NextDouble();
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