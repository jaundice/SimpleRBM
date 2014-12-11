using Cudafy;
using Cudafy.Host;

namespace SimpleRBM.Cuda
{
    public static partial class MatrixEx
    {
        public static Matrix2D<double> Multiply(this Matrix2D<double> self, double scalar)
        {
            Matrix2D<double> output = self.GPU.AllocateAndSet<double>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.MultiplyScalarD, self.Matrix, scalar, output.Matrix);
            return output;
        }

        public static Matrix2D<double> Multiply(this Matrix2D<double> self, Matrix2D<double> other)
        {
            Matrix2D<double> result = self.GPU.AllocateAndSet<double>(self.GetLength(0), other.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(result, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.MultiplyD, self.Matrix, other.Matrix, result.Matrix);
            return result;
        }
        public static void Increment(this Matrix2D<double> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.IncrementD, self.Matrix);
        }

        public static void Identity(this Matrix2D<double> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.IdentityD, self.Matrix);
        }

        public static void InsertValuesFrom(this Matrix2D<double> self, int mPos, int nPos, Matrix2D<double> source,
            int mSize = 0, int nSize = 0)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(source, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.InsertValuesFromD, self.Matrix, mPos, nPos, source.Matrix, mSize,
                nSize);
        }

        public static void UpdateValuesAlongAxis(this Matrix2D<double> self, int index, double value, Axis axis)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.UpdateValueAlongAxisD, self.Matrix, index, value,
                axis == Axis.Row ? Matrix2DCudaF.TRUE : Matrix2DCudaF.FALSE);
        }

        public static Matrix2D<double> Logistic(this Matrix2D<double> self)
        {
            Matrix2D<double> res = self.GPU.AllocateAndSet<double>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.Logistic, self.Matrix, res.Matrix);
            return res;
        }


        public static Matrix2D<double> GreaterThan(this Matrix2D<double> self, Matrix2D<double> other)
        {
            Matrix2D<double> res = self.GPU.AllocateAndSet<double>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaD.GreaterThanD, self.Matrix, other.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<double> SubMatrix(this Matrix2D<double> self, int startRow, int startCol, int numRows = 0,
            int numCols = 0)
        {
            numRows = numRows != 0 ? numRows : self.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : self.GetLength(1) - startCol;

            Matrix2D<double> res = self.GPU.AllocateAndSet<double>(numRows, numCols);

            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(numRows, numCols, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaD.SubMatrixD, self.Matrix, startRow, startCol, numRows, numCols,
                res.Matrix);

            return res;
        }

        public static Matrix2D<double> Transpose(this Matrix2D<double> self)
        {
            Matrix2D<double> res = self.GPU.AllocateAndSet<double>(self.GetLength(1), self.GetLength(0));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(res, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaD.TransposeD, self.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<double> Subtract(this Matrix2D<double> self, Matrix2D<double> other)
        {
            Matrix2D<double> res = self.GPU.AllocateAndSet<double>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaD.SubtractD, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }

        public static Matrix2D<double> Add(this Matrix2D<double> self, Matrix2D<double> other)
        {
            Matrix2D<double> res = self.GPU.AllocateAndSet<double>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaD.AddD, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }

        public static Matrix2D<double> Pow(this Matrix2D<double> self, double power)
        {
            Matrix2D<double> res = self.GPU.AllocateAndSet<double>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaD.PowD, self.Matrix, power, res.Matrix);
            return res;
        }

        public static Matrix2D<double> Upload(GPGPU gpu, double[,] source)
        {
            Matrix2D<double> tempSrcData = gpu.AllocateAndSet<double>(source.GetLength(0), source.GetLength(1));
            gpu.CopyToDevice(source, tempSrcData);
            return tempSrcData;
        }

        public static double[,] CopyLocal(this Matrix2D<double> self)
        {
            var res = new double[self.GetLength(0), self.GetLength(1)];
            self.GPU.CopyFromDevice(self.Matrix, res);
            return res;
        }

        public static void Ones(this Matrix2D<double> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.OnesD, self.Matrix);
        }

        public static void Zeros(this Matrix2D<double> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.ZerosD, self.Matrix);
        }

        public static void Fill(this Matrix2D<double> self, double value)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.FillD, self.Matrix, value);
        }

        public static void InsertValuesFromRowOrColumn(this Matrix2D<double> self, Matrix2D<double> source, int length,
            Axis axis, int mPos, int nPos)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(length, 1, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaD.InsertValuesFromRowOrColumnD, self.Matrix, source.Matrix, length,
                axis == Axis.Column ? Matrix2DCudaF.TRUE : Matrix2DCudaF.FALSE, mPos, nPos);
        }
    }
}