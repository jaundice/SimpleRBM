using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Cuda;

namespace SimpleRBM.Test
{
    [TestClass]
    public class LinearAlgebra : CudaTestBase
    {

        [TestMethod]
        public void CudaGuassian()
        {
            using (var guassian = _dev.GuassianDistribution(_rand, 500, 500, 0.0))
            {
                PrintArray(guassian.CopyLocal());
            }
        }


        [TestMethod]
        public void RepMatRows()
        {
            double[,] source = new double[1, 7] { { 1, 2, 3, 4, 5, 6, 7 } };

            double[,] res;

            using (var data = _dev.Upload(source))
            using (var repl = data.RepMatRows(1000))
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
            using (var repl = data.RepMatCols(1000))
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
            float ret;
            using (var data = _dev.Upload(src))
            {
                ret = data.Sum();
            }
            Console.WriteLine(ret);
            PrintArray(src);
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
            using (var bin = _dev.GuassianDistribution(_rand, 300, 400, 0.0))
            {
                bin.ToBinary();

                using (
                    var trans = bin.Transpose())
                {
                    PrintArray(bin.CopyLocal());
                    Console.WriteLine();
                    PrintArray(trans.CopyLocal());
                }
            }
        }

        [TestMethod]
        public void Fill()
        {
            using (var m = _dev.AllocateAndSet<double>(2000, 500))
            {
                m.Fill(3);
                PrintArray(m.CopyLocal());
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

        public static void PrintArray<T>(T[,] ret)
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