using System;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.BLAS.Types;
using GASS.CUDA.BLAS;
using SimpleRBM.Cuda.CudaMatrix;
using TElement = System.Double;

namespace SimpleRBM.Cuda
{
    public static partial class MatrixEx
    {

        public static TElement Sum(this Matrix2D<TElement> self)
        {
            using (var as1double = self.Cast1D())
            {
                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                var v = blas.ASUM(as1double.Matrix);
                self.GPU.SetCurrentContext();
                self.GPU.Synchronize();

                return v;
            }
        }

        public static Matrix2D<TElement> Multiply(this Matrix2D<TElement> self, TElement scalar)
        {
            var n = self.CloneOnDevice();
            n.MultiplyInPlace(scalar);
            self.GPU.SetCurrentContext();
            self.GPU.Synchronize();

            return n;
        }

        public static Matrix2D<TElement> MultiplyInPlace(this Matrix2D<TElement> self, TElement scalar)
        {
            using (var as1D = self.Cast1D())
            {
                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                blas.SCAL(scalar, as1D.Matrix);
            }
            self.GPU.SetCurrentContext();
            self.GPU.Synchronize();

            return self;
        }


        public static Matrix2D<TElement> Multiply(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            var blas = CudaToolsRegistry.GetBlas(self.GPU);
            int m = self.GetLength(0);
            int k = self.GetLength(1);
            int n = other.GetLength(1);
            var working = self.GPU.AllocateNoSet<TElement>(m, n);

            using (var otherAsSingle = other.Cast1D())
            using (var selfAsSingle = self.Cast1D())
            using (var workingAsSingle = working.Cast1D())
            {
                blas.GEMM(n, k, m, 1, otherAsSingle.Matrix, selfAsSingle.Matrix, 1, workingAsSingle.Matrix, cublasOperation.N, cublasOperation.N);
            }
            self.GPU.SetCurrentContext();
            self.GPU.Synchronize();

            return working;
        }

        public enum MultiplyPreOp
        {
            Normal,
            Transpose
        }

            //public static Matrix1D<TElement> ToSingleRank(this Matrix2D<TElement> self, out int stride)
        //{
        //    Matrix1D<TElement> result = self.GPU.AllocateNoSet<TElement>(self.GetLength(0) * self.GetLength(1));
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
        //    self.GPU.Launch(grid, block, Matrix2DCuda.ToSingleRankD, self.Matrix, result.Matrix);
        //    stride = self.GetLength(1);
        //    return result;
        //}

        //public static Matrix2D<TElement> ToDoubleRank(this Matrix1D<TElement> self, int stride)
        //{

        //    Matrix2D<TElement> result = self.GPU.AllocateNoSet<TElement>(self.GetLength(0) / stride, stride);
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(result, out grid, out block);
        //    self.GPU.Launch(grid, block, Matrix2DCuda.ToDoubleRankD, self.Matrix, result.Matrix);
        //    return result;
        //}


        public static void Increment(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.IncrementD, self.Matrix);
        }

        public static void Identity(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.IdentityD, self.Matrix);
        }

        public static void InsertValuesFrom(this Matrix2D<TElement> self, int mPos, int nPos, Matrix2D<TElement> source,
            int mSize = 0, int nSize = 0)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(source, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.InsertValuesFromD, self.Matrix, mPos, nPos, source.Matrix, mSize,
                nSize);
        }

        public static void UpdateValuesAlongAxis(this Matrix2D<TElement> self, int index, TElement value, Axis axis)
        {
            dim3 grid, block;
            if (axis == Axis.Row)
            {
                ThreadOptimiser.Instance.GetStrategy(self.GetLength(1), 1, out grid, out block);
            }
            else
            {
                ThreadOptimiser.Instance.GetStrategy(1, self.GetLength(0), out grid, out block);
            }
            //ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.UpdateValueAlongAxisD, self.Matrix, index, value,
                axis == Axis.Row ? Matrix2DCuda.TRUE : Matrix2DCuda.FALSE);
        }

        public static Matrix2D<TElement> Logistic(this Matrix2D<TElement> self)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.LogisticD, self.Matrix, res.Matrix);
            return res;
        }

        public static Matrix2D<TElement> LogisticInPlace(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.LogisticInPlaceD, self.Matrix);
            return self;
        }

        public static Matrix2D<TElement> TanhInPlace(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.TanhInPlaceD, self.Matrix);
            return self;
        }

        public static Matrix2D<TElement> SoftPlusInPlace(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.SoftPlusInPlaceD, self.Matrix);
            return self;
        }

        public static Matrix2D<TElement> Exponents(this Matrix2D<TElement> self)
        {
            var res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, ActivationFunctionsCuda.ExponentsD, self.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<TElement> MaxRowValues(this Matrix2D<TElement> self)
        {
            var res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), 1);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.MaximumElementValueRowWiseD, self.Matrix, res.Matrix);
            return res;
        }

        public static Matrix2D<TElement> DivideElements(this Matrix2D<TElement> self, Matrix2D<TElement> denominator)
        {
            var res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.DivideByD, self.Matrix, denominator.Matrix, res.Matrix);

            return res;

        }

        public static Matrix2D<TElement> SoftMax(this Matrix2D<TElement> self)
        {
            using (var max = self.MaxRowValues())
            using (var neg = max.RepMatCols(self.GetLength(1)))
            using (var delta = self.Subtract(neg))
            using (var exp = delta.Exponents())
            using (var summedExp = exp.SumRows())
            using (var tiledSummed = summedExp.RepMatCols(self.GetLength(1)))
            {
                var res = exp.DivideElements(tiledSummed);
                return res;
            }



            //using (Matrix2D<TElement> exponents = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1)))
            //{
            //    dim3 grid, block;
            //    ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);
            //    self.GPU.Launch(grid, block, ActivationFunctionsCuda.ExponentsD, self.Matrix, exponents.Matrix);

            //    using (Matrix2D<TElement> scales = exponents.SumRows())
            //    {
            //        //ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);

            //        //self.GPU.Launch(grid, block, Matrix2DCuda.SumMatrixRowsD, exponents.Matrix, scales.Matrix);
            //        ////todo could do this in place and save an allocation

            //        Matrix2D<TElement> result = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            //        ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //        self.GPU.Launch(grid, block, Matrix2DCuda.DivideByD, exponents.Matrix, scales.Matrix, result.Matrix);

            //        return result;
            //    }
            //}
        }


        public static Matrix2D<TElement> GreaterThan(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.GreaterThanD, self.Matrix, other.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<TElement> GreaterThanLinear(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.GreaterThanLinearD, self.Matrix, other.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<TElement> SubMatrix(this Matrix2D<TElement> self, int startRow, int startCol, int numRows = 0,
            int numCols = 0)
        {
            numRows = numRows != 0 ? numRows : self.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : self.GetLength(1) - startCol;

            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(numRows, numCols);

            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(numRows, numCols, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.SubMatrixD, self.Matrix, startRow, startCol, numRows, numCols,
                res.Matrix);

            return res;
        }

        public static Matrix2D<TElement> Transpose(this Matrix2D<TElement> self)
        {
            Matrix2D<TElement> res = self.GPU.AllocateAndSet<TElement>(self.GetLength(1), self.GetLength(0));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(res, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.TransposeD, self.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<TElement> Subtract(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            Matrix2D<TElement> res = self.CloneOnDevice();

            res.SubtractInPlace(other);
            return res;
        }

        public static Matrix2D<TElement> SubtractInPlace(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {

            using (var self1d = self.Cast1D())
            using (var other1d = other.Cast1D())
            {

                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                blas.AXPY(-1, other1d.Matrix, self1d.Matrix);
            }
            return self;
        }

        public static Matrix2D<TElement> Add(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            Matrix2D<TElement> res = self.CloneOnDevice();
            res.AddInPlace(other);
            return res;
        }


        public static Matrix2D<TElement> AddInPlace(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            using (var self1d = self.Cast1D())
            using (var other1d = other.Cast1D())
            {

                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                blas.AXPY(1, other1d.Matrix, self1d.Matrix);
            }
            return self;
        }

        public static Matrix2D<TElement> UpdateWithMomentum(this Matrix2D<TElement> self, Matrix2D<TElement> other,
            TElement momentum)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.UpdateWithMomentumD, self.Matrix, other.Matrix, res.Matrix,
                momentum);
            return res;
        }

        public static Matrix2D<TElement> UpdateWithMomentumInPlace(this Matrix2D<TElement> self, Matrix2D<TElement> other, TElement momentum)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.UpdateWithMomentumInPlaceD, self.Matrix, other.Matrix, momentum);
            return self;
        }

        public static Matrix2D<TElement> Pow(this Matrix2D<TElement> self, TElement power)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.PowD, self.Matrix, power, res.Matrix);
            return res;
        }

        public static Matrix2D<TElement> PowInPlace(this Matrix2D<TElement> self, TElement power)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.PowInPlaceD, self.Matrix, power);
            return self;
        }

        //public static Matrix2D<TElement> Upload(this GPGPU gpu, TElement[,] source)
        //{
        //    Matrix2D<TElement> tempSrcData = gpu.AllocateNoSet<TElement>(source.GetLength(0), source.GetLength(1));
        //    gpu.CopyToDevice(source, tempSrcData);
        //    return tempSrcData;
        //}

        public static TElement[,] CopyLocal(this Matrix2D<TElement> self)
        {
            var res = new TElement[self.GetLength(0), self.GetLength(1)];
            self.GPU.CopyFromDevice(self.Matrix, res);
            return res;
        }

        public static TElement[] CopyLocal(this Matrix1D<TElement> self)
        {
            var res = new TElement[self.GetLength(0)];
            self.GPU.CopyFromDevice(self.Matrix, res);
            return res;
        }

        public static void Ones(this Matrix2D<TElement> self)
        {
            self.Fill(1f);
            //dim3 grid, block;
            //ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //self.GPU.Launch(grid, block, Matrix2DCuda.OnesD, self.Matrix);
        }

        public static void Zeros(this Matrix2D<TElement> self)
        {
            self.Set();
            //dim3 grid, block;
            //ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //self.GPU.Launch(grid, block, Matrix2DCuda.ZerosD, self.Matrix);
        }

        public static void Fill(this Matrix2D<TElement> self, TElement value)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.FillD, self.Matrix, value);
        }

        public static void ToBinary(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.ToBinaryD, self.Matrix);
        }

        public static void InsertValuesFromRowOrColumn(this Matrix2D<TElement> self, Matrix2D<TElement> source,
            Axis axis, int mPos, int nPos)
        {
            dim3 grid, block;
            int length = axis == Axis.Row ? source.GetLength(1) : source.GetLength(0);

            ThreadOptimiser.Instance.GetStrategy(length, 1, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.InsertValuesFromRowOrColumnD, self.Matrix, source.Matrix, length,
                axis == Axis.Column ? Matrix2DCuda.TRUE : Matrix2DCuda.FALSE, mPos, nPos);
        }

        //public static Matrix2D<TElement> RepMatRows(this Matrix2D<TElement> self, int clones)
        //{
        //    if (self.GetLength(0) != 1)
        //        throw new Exception();

        //    var ret = self.GPU.AllocateNoSet<TElement>(clones, self.GetLength(1));
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(ret, out grid, out block);

        //    self.GPU.Launch(grid, block, Matrix2DCuda.RepMatRowsD, self.Matrix, ret.Matrix);



        //    return ret;
        //}

        public static Matrix2D<TElement> RepMatRows(this Matrix2D<TElement> self, int clones)
        {
            if (self.GetLength(0) != 1)
                throw new Exception();

            var ret = self.GPU.AllocateAndSet<TElement>(clones, self.GetLength(1));

            using (var ones = self.GPU.AllocateNoSet<TElement>(clones, self.GetLength(1)))
            using (var ones1d = ones.Cast1D())
            using (var self1d = self.Cast1D())
            using (var ret1D = ret.Cast1D())
            {
                ones.Fill(1.0);
                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                blas.GER(ret.GetLength(1), ret.GetLength(0), 1, self1d.Matrix, ones1d.Matrix, ret1D.Matrix);
            }


            return ret;
        }

        public static Matrix2D<TElement> RepMatCols(this Matrix2D<TElement> self, int clones)
        {
            if (self.GetLength(1) != 1)
                throw new Exception();

            var ret = self.GPU.AllocateAndSet<TElement>(self.GetLength(0), clones);

            using (var ones = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), clones))
            using (var ones1d = ones.Cast1D())
            using (var self1d = self.Cast1D())
            using (var ret1D = ret.Cast1D())
            {
                ones.Fill(1.0);
                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                blas.GER(ret.GetLength(1), ret.GetLength(0), 1, ones1d.Matrix, self1d.Matrix, ret1D.Matrix);
            }


            return ret;
        }


        //public static Matrix2D<TElement> RepMatCols(this Matrix2D<TElement> self, int clones)
        //{
        //    if (self.GetLength(1) != 1)
        //        throw new Exception();

        //    var ret = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), clones);
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(ret, out grid, out block);

        //    self.GPU.Launch(grid, block, Matrix2DCuda.RepMatColsD, self.Matrix, ret.Matrix);



        //    return ret;
        //}

        //public static Matrix2D<TElement> SumRows(this Matrix2D<TElement> self)
        //{
        //    var ret = self.GPU.AllocateAndSet<TElement>(self.GetLength(0), 1);
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);

        //    self.GPU.Launch(grid, block, Matrix2DCuda.SumMatrixRowsD, self.Matrix, ret.Matrix);
        //    return ret;
        //}

        //public static Matrix2D<TElement> SumColumns(this Matrix2D<TElement> self)
        //{
        //    var ret = self.GPU.AllocateAndSet<TElement>(1, self.GetLength(1));
        //    dim3 grid, block;
        //    ThreadOptimiser.Instance.GetStrategy(self.GetLength(1), 1, out grid, out block);

        //    self.GPU.Launch(grid, block, Matrix2DCuda.SumMatrixColumnsD, self.Matrix, ret.Matrix);
        //    return ret;
        //}

        public static Matrix2D<TElement> SumRows(this Matrix2D<TElement> self)
        {
            var ret = self.GPU.AllocateAndSet<TElement>(self.GetLength(0), 1);

            using (var ones = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), 1))
            using (var ones1d = ones.Cast1D())
            using (var self1d = self.Cast1D())
            using (var ret1D = ret.Cast1D())
            {
                ones.Fill(1.0);
                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                blas.GEMV(self.GetLength(1), self.GetLength(0), 1, self1d.Matrix, ones1d.Matrix, 1, ret1D.Matrix, cublasOperation.T);
            }


            return ret;
        }

        public static Matrix2D<TElement> SumColumns(this Matrix2D<TElement> self)
        {
            var ret = self.GPU.AllocateAndSet<TElement>(1, self.GetLength(1));

            using (var ones = self.GPU.AllocateNoSet<TElement>(1, self.GetLength(1)))
            using (var ones1d = ones.Cast1D())
            using (var self1d = self.Cast1D())
            using (var ret1D = ret.Cast1D())
            {
                ones.Fill(1.0f);
                var blas = CudaToolsRegistry.GetBlas(self.GPU);
                blas.GEMV(self.GetLength(1), self.GetLength(0), 1, self1d.Matrix, ones1d.Matrix, 1, ret1D.Matrix, cublasOperation.N);
            }


            return ret;
        }
    }


}