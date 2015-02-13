﻿using System;
using Cudafy;
using Cudafy.Host;
using TElement = System.Double;

namespace SimpleRBM.Cuda
{
    public static partial class MatrixEx
    {

        public static TElement Sum(this Matrix2D<TElement> self)
        {
            using (var cols = self.SumColumns())
            using (var rows = cols.SumRows())
            {
                return rows.CopyLocal()[0, 0];
            }
        }


        public static Matrix2D<TElement> Multiply(this Matrix2D<TElement> self, TElement scalar)
        {
            Matrix2D<TElement> output = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.MultiplyScalarD, self.Matrix, scalar, output.Matrix);
            return output;
        }

        public static void MultiplyInPlace(this Matrix2D<TElement> self, TElement scalar)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCuda.MultiplyScalarInPlaceD, self.Matrix, scalar);
        }

        public static Matrix2D<TElement> Multiply(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            Matrix2D<TElement> result = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), other.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), other.GetLength(1), out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.MultiplyD, self.Matrix, other.Matrix, result.Matrix);
            return result;
        }

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

        public static void LogisticInPlace(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.LogisticInPlaceD, self.Matrix);
        }

        public static void TanhInPlace(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.TanhInPlaceD, self.Matrix);
        }

        public static void SoftPlusInPlace(this Matrix2D<TElement> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.SoftPlusInPlaceD, self.Matrix);
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
                return delta.DivideElements(tiledSummed);
            }



            //using (Matrix2D<TElement> exponents = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1)))
            //{
            //    dim3 grid, block;
            //    ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);
            //    self.GPU.Launch(grid, block, ActivationFunctionsCuda.ExponentsF, self.Matrix, exponents.Matrix);

            //    using (Matrix2D<TElement> scales = exponents.SumRows())
            //    {
            //        //ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);

            //        //self.GPU.Launch(grid, block, Matrix2DCuda.SumMatrixRowsF, exponents.Matrix, scales.Matrix);
            //        ////todo could do this in place and save an allocation

            //        Matrix2D<TElement> result = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            //        ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //        self.GPU.Launch(grid, block, Matrix2DCuda.DivideByF, exponents.Matrix, scales.Matrix, result.Matrix);

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
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(1), self.GetLength(0));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(res, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.TransposeD, self.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<TElement> Subtract(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.SubtractD, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }

        public static void SubtractInPlace(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.SubtractInPlaceD, self.Matrix, other.Matrix);
        }

        public static Matrix2D<TElement> Add(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.AddD, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }

        public static void AddInPlace(this Matrix2D<TElement> self, Matrix2D<TElement> other)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.AddInPlaceD, self.Matrix, other.Matrix);
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

        public static void UpdateWithMomentumInPlace(this Matrix2D<TElement> self, Matrix2D<TElement> other, TElement momentum)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.UpdateWithMomentumInPlaceD, self.Matrix, other.Matrix, momentum);
        }

        public static Matrix2D<TElement> Pow(this Matrix2D<TElement> self, TElement power)
        {
            Matrix2D<TElement> res = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.PowD, self.Matrix, power, res.Matrix);
            return res;
        }

        public static void PowInPlace(this Matrix2D<TElement> self, TElement power)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.PowInPlaceD, self.Matrix, power);
        }

        public static Matrix2D<TElement> Upload(this GPGPU gpu, TElement[,] source)
        {
            Matrix2D<TElement> tempSrcData = gpu.AllocateNoSet<TElement>(source.GetLength(0), source.GetLength(1));
            gpu.CopyToDevice(source, tempSrcData);
            return tempSrcData;
        }

        public static TElement[,] CopyLocal(this Matrix2D<TElement> self)
        {
            var res = new TElement[self.GetLength(0), self.GetLength(1)];
            self.GPU.CopyFromDevice(self.Matrix, res);
            return res;
        }

        public static void Ones(this Matrix2D<TElement> self)
        {
            self.Fill(1f);
            //dim3 grid, block;
            //ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //self.GPU.Launch(grid, block, Matrix2DCuda.OnesF, self.Matrix);
        }

        public static void Zeros(this Matrix2D<TElement> self)
        {
            self.Set();
            //dim3 grid, block;
            //ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //self.GPU.Launch(grid, block, Matrix2DCuda.ZerosF, self.Matrix);
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

        public static Matrix2D<TElement> RepMatRows(this Matrix2D<TElement> self, int clones)
        {
            if (self.GetLength(0) != 1)
                throw new Exception();

            var ret = self.GPU.AllocateNoSet<TElement>(clones, self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(ret, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.RepMatRowsD, self.Matrix, ret.Matrix);



            return ret;
        }


        public static Matrix2D<TElement> RepMatCols(this Matrix2D<TElement> self, int clones)
        {
            if (self.GetLength(1) != 1)
                throw new Exception();

            var ret = self.GPU.AllocateNoSet<TElement>(self.GetLength(0), clones);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(ret, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.RepMatColsD, self.Matrix, ret.Matrix);



            return ret;
        }

        public static Matrix2D<TElement> SumRows(this Matrix2D<TElement> self)
        {
            var ret = self.GPU.AllocateAndSet<TElement>(self.GetLength(0), 1);
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.SumMatrixRowsD, self.Matrix, ret.Matrix);
            return ret;
        }

        public static Matrix2D<TElement> SumColumns(this Matrix2D<TElement> self)
        {
            var ret = self.GPU.AllocateAndSet<TElement>(1, self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self.GetLength(1), 1, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCuda.SumMatrixColumnsD, self.Matrix, ret.Matrix);
            return ret;
        }
    }
}