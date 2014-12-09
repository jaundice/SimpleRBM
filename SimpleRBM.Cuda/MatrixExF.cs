using Cudafy;
using Cudafy.Host;
using SimpleRBM.Common;

namespace SimpleRBM.Cuda
{
    public static partial class MatrixEx
    {
        public static Matrix2D<float> Multiply(this Matrix2D<float> self, float scalar)
        {
            Matrix2D<float> output = self.GPU.AllocateAndSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.MultiplyScalar, self.Matrix, scalar, output.Matrix);
            return output;
        }

        public static Matrix2D<float> Multiply(this Matrix2D<float> self, Matrix2D<float> other)
        {
            Matrix2D<float> result = self.GPU.AllocateAndSet<float>(self.GetLength(0), other.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self.GetLength(0), other.GetLength(1), out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.Multiply, self.Matrix, other.Matrix, result.Matrix);
            return result;
        }

        public static void InsertValuesFrom(this Matrix2D<float> self, int mPos, int nPos, Matrix2D<float> source,
            int mSize = 0, int nSize = 0)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(source, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.InsertValuesFrom, self.Matrix, mPos, nPos, source.Matrix, mSize,
                nSize);
        }

        public static void UpdateValuesAlongAxis(this Matrix2D<float> self, int index, float value, Axis axis)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.UpdateValueAlongAxis, self.Matrix, index, value,
                axis == Axis.Row ? Matrix2DCudaF.TRUE : Matrix2DCudaF.FALSE);
        }

        public static Matrix2D<float> Logistic(this Matrix2D<float> self)
        {
            var res = self.GPU.AllocateAndSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, ActivationFunctionsCuda.LogisticF, self.Matrix, res.Matrix);
            return res;
        }


        public static Matrix2D<float> GreaterThan(this Matrix2D<float> self, Matrix2D<float> other)
        {
            var res = self.GPU.AllocateAndSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.GreaterThan, self.Matrix, other.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<float> SubMatrix(this Matrix2D<float> self, int startRow, int startCol, int numRows = 0,
            int numCols = 0)
        {
            numRows = numRows != 0 ? numRows : self.GetLength(0) - startRow;
            numCols = numCols != 0 ? numCols : self.GetLength(1) - startCol;

            var res = self.GPU.AllocateAndSet<float>(numRows, numCols);

            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(numRows, numCols, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.SubMatrix, self.Matrix, startRow, startCol, numRows, numCols, res.Matrix);

            return res;
        }

        public static Matrix2D<float> Transpose(this Matrix2D<float> self)
        {
            var res = self.GPU.AllocateAndSet<float>(self.GetLength(1), self.GetLength(0));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(res, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.Transpose, self.Matrix, res.Matrix);

            return res;
        }

        public static Matrix2D<float> Subtract(this Matrix2D<float> self, Matrix2D<float> other)
        {
            var res = self.GPU.AllocateAndSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.Subtract, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }

        public static Matrix2D<float> Add(this Matrix2D<float> self, Matrix2D<float> other)
        {
            var res = self.GPU.AllocateAndSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.Add, self.Matrix, other.Matrix, res.Matrix);
            return res;
        }

        public static Matrix2D<float> Pow(this Matrix2D<float> self, float power)
        {
            var res = self.GPU.AllocateAndSet<float>(self.GetLength(0), self.GetLength(1));
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);

            self.GPU.Launch(grid, block, Matrix2DCudaF.Pow, self.Matrix, power, res.Matrix);
            return res;
        }

        public static Matrix2D<float> Upload(GPGPU gpu, float[,] source)
        {
            Matrix2D<float> tempSrcData = gpu.AllocateAndSet<float>(source.GetLength(0), source.GetLength(1));
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
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.Ones, self.Matrix);
        }

        public static void Zeros(this Matrix2D<float> self)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(self, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.Zeros, self.Matrix);
        }

        public static void InsertValuesFromRowOrColumn(this Matrix2D<float> self, Matrix2D<float> source, int length,
            Axis axis, int mPos, int nPos)
        {
            dim3 grid, block;
            ThreadOptimiser.Instance.GetStrategy(length, 1, out grid, out block);
            self.GPU.Launch(grid, block, Matrix2DCudaF.InsertValuesFromRowOrColumn, self.Matrix, source.Matrix, length, axis == Axis.Column ? Matrix2DCudaF.TRUE : Matrix2DCudaF.FALSE, mPos, nPos);

        }
    }

    public enum Axis
    {
        Row,
        Column
    }

  
}