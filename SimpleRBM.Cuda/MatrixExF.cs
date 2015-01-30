using Cudafy;
using Cudafy.Host;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public static partial class MatrixEx
    {
        public static Matrix2D<float> Multiply(this Matrix2D<float> self, float scalar)
        {
            Matrix2D<float> output = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.MultiplyScalarF, self.Matrix, scalar, output.Matrix);
            return output;
        }

        public static void MultiplyInPlace(this Matrix2D<float> self, float scalar)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.MultiplyScalarInPlaceF, self.Matrix, scalar);
        }

        public static Matrix2D<float> Multiply(this Matrix2D<float> self, Matrix2D<float> other)
        {
            Matrix2D<float> result = self.GPU.AllocateNoSet<float>(self.GetLength(0), other.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), other.GetLength(1), out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.MultiplyF, self.Matrix, other.Matrix, result.Matrix);
            return result;
        }
        public static void Increment(this Matrix2D<float> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.IncrementF, self.Matrix);
        }

        public static void Identity(this Matrix2D<float> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.IdentityF, self.Matrix);
        }
        public static void InsertValuesFrom(this Matrix2D<float> self, int mPos, int nPos, Matrix2D<float> source,
            int mSize = 0, int nSize = 0)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(source, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.InsertValuesFromF, self.Matrix, mPos, nPos, source.Matrix, mSize,
                nSize);
        }

        public static void UpdateValuesAlongAxis(this Matrix2D<float> self, int index, float value, Axis axis)
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
            self.GPU.Launch(grid, block, Matrix2DCudaF.UpdateValueAlongAxisF, self.Matrix, index, value,
                axis == Axis.Row ? Matrix2DCudaF.TRUE : Matrix2DCudaF.FALSE);
        }

        public static Matrix2D<float> Logistic(this Matrix2D<float> self)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.LogisticF, self.Matrix, res.Matrix);
            return res;
        }

        public static void LogisticInPlace(this Matrix2D<float> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.LogisticInPlaceF, self.Matrix);

        }


        public static Matrix2D<float> SoftMax(this Matrix2D<float> self)
        {
            using (var exponents = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1)))
            {
                dim3 grid, block;
                ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);
                self.GPU.Launch(grid, block, ActivationFunctionsCuda.LogSumOfExponentsF, self.Matrix, exponents.Matrix);

                using (var scales = self.GPU.AllocateNoSet<float>(self.GetLength(0), 1))
                {
                    ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), 1, out grid, out block);

                    self.GPU.Launch(grid, block, Matrix2DCudaF.SumMatrixRowsF, exponents.Matrix, scales.Matrix);
                    //todo could do this in place and save an allocation
                    var result = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
                    ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
                    self.GPU.Launch(grid, block, Matrix2DCudaF.DivideByF, exponents.Matrix, scales.Matrix, result.Matrix);

                    return result;
                }
            }

        }


        public static Matrix2D<float> GreaterThan(this Matrix2D<float> self, Matrix2D<float> other)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.GreaterThanF, self.Matrix, other.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<float> GreaterThanLinear(this Matrix2D<float> self, Matrix2D<float> other)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.GreaterThanLinearF, self.Matrix, other.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<float> SubMatrix(this Matrix2D<float> self, int startRow, int startCol, int numRows = 0,
            int numCols = 0)
        {
            numRows = numRows != 0 ? numRows : self.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : self.GetLength(1) - startCol;

            var res = self.GPU.AllocateNoSet<float>(numRows, numCols);

            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(numRows, numCols, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.SubMatrixF, self.Matrix, startRow, startCol, numRows, numCols, res.Matrix);

            return res;
        }

        public static Matrix2D<float> Transpose(this Matrix2D<float> self)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(1), self.GetLength(0));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(res, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.TransposeF, self.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<float> Subtract(this Matrix2D<float> self, Matrix2D<float> other)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.SubtractF, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }
        public static void SubtractInPlace(this Matrix2D<float> self, Matrix2D<float> other)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.SubtractInPlaceF, self.Matrix, other.Matrix);
        }
        public static Matrix2D<float> Add(this Matrix2D<float> self, Matrix2D<float> other)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.AddF, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }

        public static void AddInPlace(this Matrix2D<float> self, Matrix2D<float> other)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.AddInPlaceF, self.Matrix, other.Matrix);
        }

        public static Matrix2D<float> UpdateWithMomentum(this Matrix2D<float> self, Matrix2D<float> other, float momentum)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.UpdateWithMomentumF, self.Matrix, other.Matrix, res.Matrix, momentum);
            return res;
        }

        public static void UpdateWithMomentumInPlace(this Matrix2D<float> self, Matrix2D<float> other, float momentum)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.UpdateWithMomentumInPlaceF, self.Matrix, other.Matrix, momentum);
        }

        public static Matrix2D<float> Pow(this Matrix2D<float> self, float power)
        {
            var res = self.GPU.AllocateNoSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.PowF, self.Matrix, power, res.Matrix);
            return res;
        }

        public static void PowInPlace(this Matrix2D<float> self, float power)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.PowInPlaceF, self.Matrix, power);
        }

        public static Matrix2D<float> Upload(this GPGPU gpu, float[,] source)
        {
            Matrix2D<float> tempSrcData = gpu.AllocateNoSet<float>(source.GetLength(0), source.GetLength(1));
            gpu.CopyToDevice(source, tempSrcData);
            return tempSrcData;
        }

        public static float[,] CopyLocal(this Matrix2D<float> self)
        {
            var res = new float[self.GetLength(0), self.GetLength(1)];
            self.GPU.CopyFromDevice(self.Matrix, res);
            return res;
        }

        public static void Ones(this Matrix2D<float> self)
        {
            self.Fill(1f);
            //dim3 grid, block;
            //ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //self.GPU.Launch(grid, block, Matrix2DCudaF.OnesF, self.Matrix);
        }

        public static void Zeros(this Matrix2D<float> self)
        {
            self.Set();
            //dim3 grid, block;
            //ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            //self.GPU.Launch(grid, block, Matrix2DCudaF.ZerosF, self.Matrix);
        }
        public static void Fill(this Matrix2D<float> self, float value)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.FillF, self.Matrix, value);
        }

        public static void ToBinaryF(this Matrix2D<float> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.ToBinaryF, self.Matrix);
        }

        public static void InsertValuesFromRowOrColumn(this Matrix2D<float> self, Matrix2D<float> source,
            Axis axis, int mPos, int nPos)
        {
            dim3 grid, block;
            var length = axis == Axis.Row ? source.GetLength(1) : source.GetLength(0);

            ThreadOptimiser.Instance.GetStrategy(length, 1, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.InsertValuesFromRowOrColumnF, self.Matrix, source.Matrix, length, axis == Axis.Column ? Matrix2DCudaF.TRUE : Matrix2DCudaF.FALSE, mPos, nPos);

        }
    }

    public enum Axis
    {
        Row,
        Column
    }

    public static partial class MatrixEx
    {
        public static void Set<T>(this Matrix1D<T> self) where T : struct
        {
            self.GPU.Set(self.Matrix);
        }

        public static void Set<T>(this Matrix2D<T> self) where T : struct
        {
            self.GPU.Set(self.Matrix);
        }
    }
}